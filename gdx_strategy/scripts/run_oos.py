"""Apply the frozen strategy to out-of-sample data. No re-optimisation.

  python scripts/run_oos.py --is data/raw/<IS file>.xlsx --oos data/raw/<OOS file>.xlsx \\
      --config results/frozen_config.json --out results/oos/

* Refuses to run if the config's sha256 differs from the one recorded at freeze
  (``frozen_config.sha256`` next to it, or --expected-sha256), or if the IS
  workbook is not the one the config was frozen on.
* IS + OOS prices are concatenated with seam checks (overlap must be identical;
  no multi-day gap). IS history is warm-up: rolling betas and gamma are live on OOS
  day 1 and keep updating through OOS; the frozen parameters never change.
* Each frozen config starts flat and may first trade at the last IS close (a
  signal formed from IS data only), so the first OOS return is traded.
* Metrics are OOS-only, plus a side-by-side IS vs OOS table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import report as rp  # noqa: E402
from src.backtest import build_signal  # noqa: E402
from src.data import SeamError, load_is_oos  # noqa: E402
from src.factor_model import ResidualCache, check_factor_set, full_sample_ols  # noqa: E402
from src.forecast import ForecastCache  # noqa: E402
from src.grid import GRID_KEYS, config_id, load_grid, run_configs, run_grid  # noqa: E402
from src.metrics import MIN_DAYS_PER_MONTH, grid_metrics, trade_list  # noqa: E402
from src.robustness import FILTERS, hard_filters  # noqa: E402

log = logging.getLogger("run_oos")
PARAMS = GRID_KEYS
COMPARE = ["sharpe", "sharpe_gross", "ann_return", "ann_vol", "sortino", "max_drawdown", "calmar", "nw_tstat",
           "hit_rate", "profit_factor", "mean_trade", "n_entries", "entries_per_month", "pct_months_with_entry",
           "min_entries_in_month", "pct_days_in_market", "avg_holding_days", "turnover_per_year",
           "pct_years_positive", "worst_year", "rolling12m_sharpe_min", "sharpe_first_half", "sharpe_second_half",
           "long_leg_pnl", "short_leg_pnl", "ann_return_ex_top1pct_days", "top10_trades_pnl_share",
           "corr_GDX", "beta_GDX", "corr_SPY", "corr_GLD", "n_days"]


class FrozenConfigError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_config(path: Path, expected: str | None = None) -> str:
    """Return the config's sha256, or raise if it doesn't match the frozen record."""
    h = sha256(path)
    if expected is None:
        side = path.with_suffix(".sha256")
        if not side.exists():
            raise FrozenConfigError(f"No recorded hash for {path.name}: expected {side} "
                                    "(written at freeze) or --expected-sha256")
        expected = side.read_text().split()[0]
    if h != expected.lower():
        raise FrozenConfigError(f"{path.name} sha256 {h[:16]}… does not match the frozen hash "
                                f"{expected[:16]}…; refusing to run a modified config")
    return h


def frozen_configs(frozen: dict) -> pd.DataFrame:
    rows = []
    for role, e in [("primary", frozen["primary"])] + [(f"alternate_{i + 1}", a)
                                                       for i, a in enumerate(frozen["alternates"])]:
        p = dict(e["params"])
        p["factors"] = check_factor_set(p["factors"])
        cid = config_id(p)
        if cid != e["config_id"]:
            raise FrozenConfigError(f"config_id mismatch: {cid} vs {e['config_id']}")
        rows.append({**p, "role": role})
    df = pd.DataFrame(rows)
    df.index = pd.Index([config_id(r) for r in rows], name="config_id")
    return df


def _code_changed_since(commit: str) -> bool | None:
    try:
        r = subprocess.run(["git", "diff", "--quiet", commit, "--", "src"], cwd=ROOT)
        return r.returncode != 0
    except Exception:  # pragma: no cover
        return None


def run(args) -> dict:
    out = Path(args.out)
    cfg_path = Path(args.config)
    cfg_hash = verify_config(cfg_path, args.expected_sha256)
    frozen = json.loads(cfg_path.read_text())
    is_path, oos_path = Path(args.is_path), Path(args.oos)
    if sha256(is_path) != frozen["data"]["sha256"]:
        raise FrozenConfigError(f"IS workbook {is_path.name} is not the file the config was frozen on "
                                f"(sha256 {frozen['data']['sha256'][:12]}…)")
    changed = _code_changed_since(frozen["git_commit"])
    if changed:
        log.warning("src/ differs from the frozen commit %s; the IS reproduction check "
                    "(is_dev_sharpe_repro_error in run_manifest.json) shows whether behaviour changed",
                    frozen["git_commit"][:10])

    fixed = frozen["fixed"]
    prices, returns, seam = load_is_oos(is_path, oos_path, rf_mode=fixed["rf_mode"],
                                        max_gap_weekdays=args.max_gap_weekdays)
    log.info("Seam: IS %s -> %s, OOS %s -> %s (%d rows); overlap dropped %d; weekdays between %d",
             seam["is_first"].date(), seam["is_last"].date(), seam["oos_first"].date(),
             seam["oos_last"].date(), seam["n_oos_rows"], seam["overlap_dropped"], seam["weekdays_between"])
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    is_last = seam["is_last"]
    configs = frozen_configs(frozen)
    cost = fixed["cost_bps"]
    grid_cfg = load_grid(ROOT / "config" / "grid.yaml")  # thresholds / k-list for diagnostics only

    # warm-up check: every frozen config has a live forecast on the last IS close
    rc = ResidualCache(returns, min_obs_frac=fixed["stage1_min_obs_frac"])
    fcache = ForecastCache(rc, window=fixed["stage2_window"], min_obs=fixed["stage2_min_obs"])
    for cid, c in configs.iterrows():
        fc = fcache.forecast(c["factors"], c["lookback"], c["m"])
        if fc.loc[is_last:, "r_hat"].iloc[:2].isna().any():
            raise RuntimeError(f"{cid}: forecast not live at the IS/OOS seam")

    # OOS run: start flat, first possible trade at the last IS close
    g = run_configs(returns, configs[PARAMS], fixed, signal_start=is_last)
    oos = np.asarray(g.dates > is_last)
    pm = grid_metrics(g, returns, oos, cost, with_trades=True)
    filt = hard_filters(pm.table, g.configs, grid_cfg["selection"])
    oos_tab = pd.concat([configs, pm.table, filt], axis=1)
    for x in grid_cfg["report"]["cost_bps"]:
        n = g.net_returns(x, mask=oos)
        oos_tab[f"sharpe_at_{x:g}bps"] = n.mean() / n.std() * np.sqrt(252)
    oos_tab.assign(factors=oos_tab["factors"].map("+".join)).to_csv(out / "oos_metrics.csv")

    # IS reference run (frozen signal start), for the side-by-side table + reproducibility check
    g_is = run_configs(returns, configs[PARAMS], fixed, signal_start=pd.Timestamp(frozen["splits"]["signal_start"]))
    d0, d1 = (pd.Timestamp(x) for x in frozen["splits"]["development"])
    v0, v1 = (pd.Timestamp(x) for x in frozen["splits"]["validation"])
    periods_is = {"development": np.asarray((g_is.dates >= d0) & (g_is.dates <= d1)),
                  "validation": np.asarray((g_is.dates >= v0) & (g_is.dates <= v1)),
                  "IS (dev+val)": np.asarray((g_is.dates >= d0) & (g_is.dates <= v1))}
    side = {}
    for name, m in periods_is.items():
        side[name] = grid_metrics(g_is, returns, m, cost, with_trades=True).table[COMPARE]
    side["OOS"] = pm.table[COMPARE]
    cmp_rows = []
    for cid in configs.index:
        for name, t in side.items():
            cmp_rows.append(pd.Series(t.loc[cid], name=(cid, name)))
    is_vs_oos = pd.DataFrame(cmp_rows)
    is_vs_oos.index = pd.MultiIndex.from_tuples(is_vs_oos.index, names=["config_id", "period"])
    is_vs_oos.to_csv(out / "is_vs_oos.csv")
    repro = {cid: float(side["development"].loc[cid, "sharpe"] - e["development"]["sharpe"])
             for cid, e in [(frozen["primary"]["config_id"], frozen["primary"])]
             + [(a["config_id"], a) for a in frozen["alternates"]]}
    if max(abs(v) for v in repro.values()) > 1e-6:
        log.warning("IS development Sharpe does not reproduce the frozen value: %s", repro)

    # primary daily detail + trades
    prim = frozen["primary"]["config_id"]
    pc = configs.loc[prim]
    fc = fcache.forecast(pc["factors"], pc["lookback"], pc["m"])
    i = g.configs.index.get_loc(prim)
    daily = pd.DataFrame({"z": fc["z"], "gamma": fc["gamma"], "gamma_t": fc["gamma_t"], "r_hat": fc["r_hat"],
                          "signal": build_signal(fc, pc["k"], pc["direction_mode"], fixed["min_edge_bps"]),
                          "pos": g.pos[i], "new_trade": g.new_trade[i], "exret_gdx": g.exret}, index=g.dates)
    daily["net"] = g.net_returns(cost, rows=[prim], mask=np.ones(len(g.dates), bool)).iloc[:, 0]
    daily.loc[daily.index >= is_last].to_csv(out / "oos_daily_primary.csv")
    tl = trade_list(g.pos[i], g.new_trade[i], g.exret, cost)
    tl["entry_date"] = g.dates[tl["entry_idx"]]
    tl[tl["entry_date"] >= is_last].drop(columns="entry_idx").to_csv(out / "oos_trades_primary.csv", index=False)

    # ---- diagnostics (same as the IS report, OOS period) ----
    oos_period = {"validation": (g.dates[oos][0], g.dates[oos][-1])}  # shaded in the OOS colour
    closes_oos = np.zeros(len(g.dates), bool)
    rows = np.flatnonzero(oos)
    closes_oos[rows - 1] = True
    exret = returns["ExRet_GDX"]
    es = {"OOS": rp.event_study(fc, exret, grid_cfg["grid"]["k"], pc["direction_mode"], closes_oos)}
    es["OOS"].to_csv(out / "oos_event_study.csv", index=False)
    f1 = rp.plot_event_study(es, figs / "event_study.png",
                             f"OOS event study, {'+'.join(pc['factors'])} L={pc['lookback']} m={pc['m']}, "
                             f"direction={pc['direction_mode']}")
    # heatmaps: diagnostic grid over OOS (never used to change the frozen config)
    full_grid = run_grid(returns, grid_cfg, signal_start=is_last)
    hm = grid_metrics(full_grid, returns, oos, cost, with_trades=False).table[["sharpe"]]
    hm = pd.concat([full_grid.configs, hm], axis=1)
    pp = frozen["primary"]["params"]
    hl = {**pp, "factors": "+".join(pp["factors"])}
    f2a = rp.plot_heatmaps(hm, "H", "k", {"lookback": pp["lookback"], "m": pp["m"],
                                           "direction_mode": pp["direction_mode"]}, figs / "heatmap_k_H.png",
                           highlight=hl, title="OOS net Sharpe over (k × H) — diagnostic only, not used for selection")
    f2b = rp.plot_heatmaps(hm, "k", "lookback", {"m": pp["m"], "H": pp["H"],
                                                  "direction_mode": pp["direction_mode"]}, figs / "heatmap_L_k.png",
                           highlight=hl, title="OOS net Sharpe over (L × k) — diagnostic only, not used for selection")
    is_shade = {"development": (seam["is_first"], is_last), **oos_period}
    f3 = rp.plot_gamma({f"{'+'.join(pc['factors'])} L={pc['lookback']} m={m}":
                        fcache.forecast(pc["factors"], pc["lookback"], m) for m in grid_cfg["grid"]["m"]},
                       is_shade, figs / "gamma.png",
                       title="Stage-2 slope γ̂_t through IS (blue) and OOS (orange)")
    F = tuple(pc["factors"])
    fs = full_sample_ols(returns.loc[:is_last], F)
    f4 = rp.plot_rolling_betas({f"L={x}": rc.betas(F, x).loc[is_last - pd.Timedelta(days=400):]
                                for x in sorted({pc["lookback"], 60, 250})},
                               figs / "rolling_betas.png", full_sample={f"beta_{f}": fs[f"Beta_{f}"] for f in F},
                               title="Stage-1 rolling betas (last ~year of IS, then OOS); dashed = IS full-sample")
    nets = g.net_returns(cost, mask=oos)
    labels = {c: f"{configs.loc[c, 'role'].upper()} {c}" for c in configs.index}
    bh = pd.Series(np.nan_to_num(g.exret[oos]), index=g.dates[oos])
    f5 = rp.plot_equity({labels[c]: nets[c] for c in configs.index}, bh, oos_period, figs / "equity.png")
    growth = rp.growth_paths(nets.rename(columns=labels), returns)
    growth.to_csv(out / "oos_growth_of_1000.csv")
    gtab = rp.growth_table(growth)
    f5b = rp.plot_growth(growth, labels[prim], oos_period, figs / "growth_of_1000.png",
                         f"OOS: growth of $1,000 invested {growth.index[0].date()} (net of {cost:g} bps)")
    months = g.dates[closes_oos].to_period("M")
    ent = {}
    for c in configs.index:
        s = pd.Series(g.new_trade[g.configs.index.get_loc(c), closes_oos], index=months).groupby(level=0).agg(["sum", "size"])
        ent[labels[c]] = s.loc[s["size"] >= MIN_DAYS_PER_MONTH, "sum"] if (s["size"] >= MIN_DAYS_PER_MONTH).any() else s["sum"]
    f6 = rp.plot_monthly_entries(ent, oos_period, figs / "monthly_entries.png")
    yearly = pd.concat({**{labels[c]: nets[c] for c in configs.index}, "buy & hold GDX": bh}, axis=1)
    ytab = yearly.groupby(yearly.index.year).sum()
    ytab.index.name = "year"
    ytab.loc["OOS ann. Sharpe"] = yearly.mean() / yearly.std() * np.sqrt(252)
    ytab.to_csv(out / "oos_yearly.csv")
    excl = rp.exclude_years_table(nets, [2008, 2020])
    costs = oos_tab[[f"sharpe_at_{x:g}bps" for x in grid_cfg["report"]["cost_bps"]]]

    prim_row = oos_tab.loc[prim]
    ok = bool(prim_row["passes"])
    banner = (f'<div class="{"ok" if ok else "warn"}"><b>OOS verdict for the primary:</b> '
              f'net Sharpe {prim_row["sharpe"]:.2f} ({prim_row["sharpe_gross"]:.2f} gross), '
              f'{int(prim_row["n_entries"])} entries, '
              f'{"passes" if ok else "fails"} the hard filters'
              + ("" if ok else f' ({", ".join(f for f in FILTERS if not prim_row[f])})')
              + f'. Frozen status was <code>{frozen["status"]}</code>.</div>')
    summary = f"""{banner}
<ul><li>Config <code>{cfg_path.name}</code> sha256 <code>{cfg_hash[:16]}…</code> verified; frozen {frozen['frozen_at']}
at commit <code>{frozen['git_commit'][:10]}</code>{' (<b>src/ has changed since</b>)' if changed else ''}.</li>
<li>IS {seam['is_first'].date()} → {is_last.date()} (warm-up only); OOS {seam['oos_first'].date()} → {seam['oos_last'].date()}
({seam['n_oos_rows']} rows, overlap dropped {seam['overlap_dropped']}, weekdays between {seam['weekdays_between']}).</li>
<li>No parameter was re-estimated: rolling betas and γ̂ update through OOS as part of the frozen rule.
Heatmaps below are diagnostics over the OOS period and were not used to change anything.</li></ul>"""
    sections = [
        ("Summary", summary),
        ("IS vs OOS (frozen configs)", rp.table_html(is_vs_oos)),
        ("OOS metrics and hard filters", rp.table_html(oos_tab.assign(factors=oos_tab["factors"].map("+".join)))),
        ("Gross vs net (OOS Sharpe at 0 / 2 / 5 bps)", rp.table_html(costs)),
        ("With / without 2008 and 2020 (OOS)", rp.table_html(excl)),
        ("1. Event study / decay curve (OOS)", rp.img_tag(f1)),
        ("2. OOS net Sharpe heatmaps (diagnostic)", rp.img_tag(f2a) + rp.img_tag(f2b)),
        ("3. Stage-2 slope γ̂ over time", rp.img_tag(f3)),
        ("4. Stage-1 rolling betas", rp.img_tag(f4)),
        ("5. Equity curve and drawdown (OOS)", rp.img_tag(f5)),
        ("Growth of $1,000 (OOS)", rp.img_tag(f5b) + rp.table_html(gtab, 2)),
        ("6. Monthly entry counts (OOS)", rp.img_tag(f6)),
        ("7. Yearly net returns (OOS)", rp.table_html(ytab)),
    ]
    rp.build_html("GDX residual-reversal strategy — out-of-sample report", sections, out / "report.html")
    manifest = {"config": str(cfg_path), "config_sha256": cfg_hash, "is_file": is_path.name,
                "is_sha256": sha256(is_path), "oos_file": oos_path.name, "oos_sha256": sha256(oos_path),
                "seam": {k: (v.isoformat()[:10] if isinstance(v, pd.Timestamp) else v) for k, v in seam.items()},
                "src_changed_since_freeze": changed, "is_dev_sharpe_repro_error": repro,
                "primary": prim, "primary_oos_sharpe": float(prim_row["sharpe"]),
                "primary_oos_passes_filters": ok}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("OOS primary %s: net Sharpe %.2f, entries %d, passes filters: %s. Report: %s",
             prim, prim_row["sharpe"], prim_row["n_entries"], ok, out / "report.html")
    return {"oos_metrics": oos_tab, "is_vs_oos": is_vs_oos, "seam": seam, "manifest": manifest,
            "grid": g, "returns": returns}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--is", dest="is_path", required=True, help="IS workbook the config was frozen on")
    ap.add_argument("--oos", required=True, help="OOS workbook (same format)")
    ap.add_argument("--config", default=str(ROOT / "results" / "frozen_config.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "oos"))
    ap.add_argument("--expected-sha256", default=None, help="override the hash recorded at freeze")
    ap.add_argument("--max-gap-weekdays", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return run(args)
    except (FrozenConfigError, SeamError) as e:
        log.error("REFUSING TO RUN: %s", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
