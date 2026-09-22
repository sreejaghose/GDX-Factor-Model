"""Full in-sample workflow.

  python scripts/run_is.py              # development only: grid, filters, ranking,
                                        # shortlist + statistical checks (no validation)
  python scripts/run_is.py --validate   # ... then evaluate the shortlist on validation
                                        # (logged), pick primary + alternates, freeze,
                                        # run the anchored walk-forward check, and report
  python scripts/run_is.py --report     # rebuild grid_results.parquet + report.html from
                                        # the existing frozen_config.json (no re-selection)

Everything is deterministic (seeded); rerunning the development stage reproduces
the same shortlist.
"""
from __future__ import annotations

import argparse
import datetime as dt
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

from src.data import check_adjusted_closes, load_dataset  # noqa: E402
from src.grid import GridResult, load_grid, run_grid  # noqa: E402
from src.factor_model import ResidualCache, full_sample_ols  # noqa: E402
from src.forecast import ForecastCache  # noqa: E402
from src.metrics import MIN_DAYS_PER_MONTH, grid_metrics  # noqa: E402
from src import report as rp  # noqa: E402
from src.robustness import (FILTERS, filter_funnel, hard_filters, make_selection_rule,  # noqa: E402
                            rank_configs, simplicity_key, statistical_checks)
from src.splits import (ValidationGate, evaluate_validation, load_splits, split_masks,  # noqa: E402
                        walk_forward)

log = logging.getLogger("run_is")
RESULTS = ROOT / "results"
PARAMS = ["factors", "lookback", "m", "k", "H", "policy", "direction_mode"]
KEY_METRICS = ["sharpe", "sharpe_gross", "ann_return", "ann_vol", "sortino", "max_drawdown", "calmar",
               "nw_tstat", "hit_rate", "profit_factor", "mean_trade", "n_entries", "entries_per_month",
               "pct_months_with_entry", "min_entries_in_month", "pct_days_in_market", "avg_holding_days",
               "turnover_per_year", "pct_years_positive", "worst_year", "worst_year_sharpe",
               "rolling12m_sharpe_min", "rolling12m_pct_positive", "sharpe_first_half", "sharpe_second_half",
               "long_leg_pnl", "short_leg_pnl", "ann_return_ex_top1pct_days", "top10_trades_pnl_share",
               "beta_GDX", "corr_GDX", "beta_SPY", "corr_SPY", "beta_GLD", "corr_GLD", "n_days"]


def _git() -> tuple[str, bool]:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "config", "scripts"],
                                             cwd=ROOT, text=True).strip())
        return h, dirty
    except Exception:  # pragma: no cover
        return "unknown", True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, dt.date)):
        return x.isoformat()[:10]
    return x


def _fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "factors" in out:
        out["factors"] = out["factors"].map("+".join)
    return out


def development(args, cfg, splits, returns, workbook):
    grid_path = RESULTS / "grid" / "is_grid.npz"
    g = run_grid(returns, cfg, include_reference=True)
    g.save(grid_path)
    masks = split_masks(g, splits)
    dev = masks["development"]
    log.info("Development: %s -> %s (%d days); validation: %s -> %s (%d days)",
             g.dates[dev][0].date(), g.dates[dev][-1].date(), dev.sum(),
             g.dates[masks["validation"]][0].date(), g.dates[masks["validation"]][-1].date(),
             masks["validation"].sum())

    ranked = rank_configs(g, returns, dev, cfg, with_trades=True)
    ranked_out = _fmt_table(ranked)
    ranked_out.to_csv(RESULTS / "grid" / "dev_ranked_all.csv.gz")
    funnel = filter_funnel(ranked[FILTERS])
    funnel.to_csv(RESULTS / "dev_filter_funnel.csv", index=False)
    log.info("Filter funnel (cumulative):\n%s", funnel.to_string(index=False))

    sel = cfg["selection"]
    other_ok = ranked[[f for f in FILTERS if f != "f_months"]].all(axis=1)
    near = ranked[~ranked["f_months"] & other_ok & (ranked["pct_months_with_entry"] >= sel["near_miss_pct_months"])]
    _fmt_table(near.head(25)[PARAMS + ["pct_months_with_entry", "min_entries_in_month", "sharpe",
                                       "nbhd_median"]]).to_csv(RESULTS / "dev_near_misses.csv")
    log.info("Near-misses (pass all but the monthly filter, >= %.0f%% months): %d",
             100 * sel["near_miss_pct_months"], len(near))

    passing = ranked[ranked["passes"]]
    if passing.empty:
        raise SystemExit("No config passes the development hard filters; stop and revisit.")
    short = passing.head(sel["shortlist_size"]).copy()

    # statistical checks on the shortlist
    n_trials = g.n_tested
    sr_trials = ranked["sharpe"].to_numpy() / np.sqrt(252)
    checks = {cid: statistical_checks(g, cid, dev, cfg["fixed"]["cost_bps"], sr_trials, n_trials,
                                      sel["stats"]) for cid in short.index}
    short = short.join(pd.DataFrame(checks).T.add_prefix("chk_"))
    for c in cfg["report"]["cost_bps"]:
        net = g.net_returns(c, rows=list(short.index), mask=dev)
        short[f"sharpe_at_{c:g}bps"] = net.mean() / net.std() * np.sqrt(252)
    cols = (["rank"] + PARAMS + ["nbhd_median", "nbhd_min", "nbhd_frac_positive", "nbhd_n"] + KEY_METRICS
            + [f"sharpe_at_{c:g}bps" for c in cfg["report"]["cost_bps"]]
            + [c for c in short.columns if c.startswith("chk_")])
    _fmt_table(short[cols]).to_csv(RESULTS / "dev_shortlist.csv")
    show = ["rank", "nbhd_median", "sharpe", "n_entries", "pct_years_positive", "ann_return_ex_top1pct_days",
            "chk_dsr", "chk_placebo_percentile", "chk_sign_flip_sharpe", "chk_boot_sharpe_lo95",
            "chk_boot_sharpe_hi95"]
    log.info("Development shortlist:\n%s", short[show].round(3).to_string())
    return g, ranked, short


def validation_and_freeze(args, cfg, splits, returns, workbook, g, ranked, short):
    sel = cfg["selection"]
    masks = split_masks(g, splits)
    dev, val = masks["development"], masks["validation"]
    cost = cfg["fixed"]["cost_bps"]
    ids = list(short.index)

    evaluate_validation(g, ids, splits, cost, reason=args.reason)  # logged gate
    vpm = grid_metrics(g, returns, val, cost, rows=ids, with_trades=True)
    vt = vpm.table
    vf = hard_filters(vt, g.configs, sel)
    vtab = pd.concat([g.configs.loc[ids, PARAMS], vt, vf], axis=1)
    for c in cfg["report"]["cost_bps"]:
        net = g.net_returns(c, rows=ids, mask=val)
        vtab[f"sharpe_at_{c:g}bps"] = net.mean() / net.std() * np.sqrt(252)
    vtab["dev_nbhd_median"] = short["nbhd_median"]
    vtab["dev_sharpe"] = short["sharpe"]
    vtab = vtab.join(simplicity_key(g.configs.loc[ids]).drop(columns="H"))
    # composite: validation Sharpe rank + development neighbourhood rank; ties -> simplicity
    vtab["composite"] = vtab["sharpe"].rank(ascending=False) + vtab["dev_nbhd_median"].rank(ascending=False)
    vtab = vtab.sort_values(["n_failed", "composite", "n_factors", "H", "policy_rank"])
    _fmt_table(vtab).to_csv(RESULTS / "validation_shortlist.csv")
    log.info("Validation (shortlist only):\n%s",
             vtab[["sharpe", "n_entries", "pct_months_with_entry", "pct_years_positive",
                   "sharpe_first_half", "sharpe_second_half", "ann_return_ex_top1pct_days",
                   "n_failed", "composite"]].round(3).to_string())

    status = "ok" if vtab["passes"].any() else "no_candidate_passed_validation_filters"
    if status != "ok":
        log.warning("No shortlisted config passes the hard filters on validation; freezing the best "
                    "by (fewest failures, composite) and flagging it.")
    chosen = list(vtab.index[: 1 + sel["n_alternates"]])
    if status == "ok":
        chosen = list(vtab.index[vtab["passes"]][: 1 + sel["n_alternates"]])

    # reference: D_stack_2x variant of each chosen config
    def d_variant(cid):
        return cid.replace("|B|", "|D|").replace("|A|", "|D|").replace("|C|", "|D|")

    def entry(cid):
        c = g.configs.loc[cid]
        dref = d_variant(cid)
        dref_m = {}
        if dref in g.configs.index:
            for name, m in (("development", dev), ("validation", val)):
                n = g.net_returns(cost, rows=[dref], mask=m).iloc[:, 0]
                dref_m[name] = float(n.mean() / n.std() * np.sqrt(252))
        return {
            "config_id": cid,
            "params": {p: (list(c[p]) if p == "factors" else c[p]) for p in PARAMS},
            "development": {**{k: short.loc[cid, k] for k in KEY_METRICS},
                            **{k: short.loc[cid, k] for k in ["nbhd_median", "nbhd_mean", "nbhd_min",
                                                               "nbhd_frac_positive", "nbhd_n", "rank"]},
                            **{f"sharpe_at_{x:g}bps": short.loc[cid, f"sharpe_at_{x:g}bps"]
                               for x in cfg["report"]["cost_bps"]},
                            "yearly_net_return": {}},
            "development_checks": {k[4:]: short.loc[cid, k] for k in short.columns if k.startswith("chk_")},
            "validation": {**{k: vtab.loc[cid, k] for k in KEY_METRICS},
                           **{f"sharpe_at_{x:g}bps": vtab.loc[cid, f"sharpe_at_{x:g}bps"]
                              for x in cfg["report"]["cost_bps"]},
                           "passes_hard_filters": vtab.loc[cid, "passes"],
                           "failed_filters": [f for f in FILTERS if not vtab.loc[cid, f]],
                           "yearly_net_return": vpm.yearly_pnl.loc[cid].to_dict()},
            "reference_D_stack_2x_sharpe": dref_m,
        }

    commit, dirty = _git()
    frozen = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "status": status,
        "data": {"file": workbook.name, "sha256": _sha256(workbook),
                 "first_date": returns.index[0], "last_date": returns.index[-1]},
        "fixed": cfg["fixed"],
        "splits": {"development": [g.dates[dev][0], g.dates[dev][-1]],
                   "validation": [g.dates[val][0], g.dates[val][-1]],
                   "signal_start": g.signal_start},
        "n_configs_tested": g.n_tested,
        "selection_rule": ("dev hard filters (" + ", ".join(FILTERS) + ") -> rank by neighbourhood median "
                           "dev net Sharpe (same categorical params, numeric params +-1 grid step) -> top "
                           f"{sel['shortlist_size']} -> validation hard filters -> composite rank of "
                           "validation Sharpe + dev neighbourhood median; ties: fewer factors, shorter H, "
                           "policy B"),
        "primary": entry(chosen[0]),
        "alternates": [entry(c) for c in chosen[1:]],
    }
    dev_yearly = grid_metrics(g, returns, dev, cost, rows=chosen, with_trades=False).yearly_pnl
    for e in [frozen["primary"], *frozen["alternates"]]:
        e["development"]["yearly_net_return"] = dev_yearly.loc[e["config_id"]].to_dict()
    (RESULTS / "frozen_config.json").write_text(json.dumps(_jsonable(frozen), indent=2))
    (RESULTS / "frozen_config.sha256").write_text(f"{_sha256(RESULTS / 'frozen_config.json')}  frozen_config.json\n")
    log.info("Frozen: primary %s; alternates %s (status: %s)", chosen[0], chosen[1:], status)

    # anchored walk-forward of the selection rule (diagnostic)
    wf_cfg = splits["walk_forward"]
    wf = walk_forward(g, make_selection_rule(returns, cfg), wf_cfg["first_test_year"],
                      wf_cfg["last_test_year"], cost)
    _fmt_table(wf["years"]).to_csv(RESULTS / "walk_forward.csv")
    s = wf["returns"]
    wf_summary = {"stability": wf["stability"],
                  "stitched_sharpe": float(s.mean() / s.std() * np.sqrt(252)),
                  "stitched_ann_return": float(s.mean() * 252),
                  "primary_in_walk_forward_years": int((wf["years"]["config_id"] == chosen[0]).sum())}
    (RESULTS / "walk_forward_summary.json").write_text(json.dumps(_jsonable(wf_summary), indent=2))
    log.info("Walk-forward:\n%s\n%s", _fmt_table(wf["years"])[
        ["config_id", "train_sharpe", "test_sharpe", "n_new_trades"]].round(3).to_string(),
        json.dumps(_jsonable(wf_summary), indent=1))
    return frozen


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

SECTION8 = ["sharpe", "sharpe_gross", "ann_return", "ann_vol", "sortino", "max_drawdown", "calmar", "nw_tstat",
            "hit_rate", "avg_win", "avg_loss", "profit_factor", "mean_trade", "n_entries", "entries_per_month",
            "pct_months_with_entry", "pct_days_in_market", "avg_holding_days", "turnover_per_year",
            "pct_years_positive", "worst_year", "rolling12m_sharpe_min", "rolling12m_pct_positive",
            "sharpe_first_half", "sharpe_second_half", "long_leg_pnl", "short_leg_pnl",
            "ann_return_ex_top1pct_days", "top10_trades_pnl_share", "corr_GDX", "beta_GDX", "corr_SPY",
            "beta_SPY", "corr_GLD", "beta_GLD"]


def grid_results_table(g, returns, masks, cost) -> pd.DataFrame:
    """Every config (incl. D reference) x period: development, validation, full, each year."""
    ValidationGate(load_splits(ROOT / "config" / "splits.yaml")["validation_log"]).check_and_log(
        list(g.configs.index),
        reason="post-freeze reporting: full-grid metrics for grid_results.parquet (selection already frozen)")
    full = masks["development"] | masks["validation"]
    periods = {"development": (masks["development"], True), "validation": (masks["validation"], True),
               "full": (full, True)}
    for y in sorted(set(g.dates[full].year)):
        periods[str(y)] = (full & np.asarray(g.dates.year == y), False)
    parts = []
    params = g.configs[PARAMS].assign(factors=g.configs["factors"].map("+".join))
    for name, (m, trades) in periods.items():
        if m.sum() < 20:
            continue
        t = grid_metrics(g, returns, m, cost, with_trades=trades).table
        parts.append(pd.concat([params, t], axis=1).assign(period=name).reset_index())
        log.info("  grid_results: %s done", name)
    return pd.concat(parts, ignore_index=True).set_index(["config_id", "period"])


def make_report(cfg, splits, returns, g, ranked, frozen):
    figs = RESULTS / "figures"
    cost = cfg["fixed"]["cost_bps"]
    masks = split_masks(g, splits)
    dev, val = masks["development"], masks["validation"]
    periods = {"development": (g.dates[dev][0], g.dates[dev][-1]),
               "validation": (g.dates[val][0], g.dates[val][-1])}
    prim = frozen["primary"]
    pp = prim["params"]
    F, L, M = tuple(pp["factors"]), pp["lookback"], pp["m"]
    finalists = [prim["config_id"]] + [a["config_id"] for a in frozen["alternates"]]
    rc = ResidualCache(returns, min_obs_frac=cfg["fixed"]["stage1_min_obs_frac"])
    fcache = ForecastCache(rc, window=cfg["fixed"]["stage2_window"], min_obs=cfg["fixed"]["stage2_min_obs"])
    exret = returns["ExRet_GDX"]

    # parquet: every config x period
    gr = grid_results_table(g, returns, masks, cost)
    f32 = gr.select_dtypes("float64").columns
    gr[f32] = gr[f32].astype("float32")
    gr.to_parquet(RESULTS / "grid_results.parquet", compression="zstd", compression_level=9)
    keep = PARAMS + ["sharpe", "sharpe_gross", "ann_return", "ann_vol", "max_drawdown", "nw_tstat", "n_entries",
                     "pct_months_with_entry", "pct_years_positive", "sharpe_first_half", "sharpe_second_half",
                     "ann_return_ex_top1pct_days", "hit_rate", "n_days"]
    summ = gr.loc[gr.index.get_level_values("period").isin(["development", "validation", "full"]), keep]
    summ.reset_index().to_csv(RESULTS / "all_configs_summary.csv.gz", index=False, float_format="%.6g")
    (RESULTS / "grid_manifest.json").write_text(json.dumps(_jsonable({
        "n_configs_total": len(g.configs), "n_selectable": g.n_tested,
        "n_reference_only": int((g.configs["policy"] == "D_stack_2x").sum()),
        "signal_start": g.signal_start, "grid_yaml_sha256": _sha256(ROOT / "config" / "grid.yaml"),
        "cost_bps": cost, "periods": ["development", "validation", "full"]}), indent=2))

    def closes(mask):
        c = np.zeros(len(g.dates), bool)
        rows = np.flatnonzero(mask)
        c[rows[rows > 0] - 1] = True
        return c

    # 1 event study
    fc = fcache.forecast(F, L, M)
    es = {name: rp.event_study(fc, exret, cfg["grid"]["k"], pp["direction_mode"], closes(m))
          for name, m in (("development", dev), ("validation", val))}
    pd.concat([d.assign(period=n) for n, d in es.items()]).to_csv(RESULTS / "event_study.csv", index=False)
    f1 = rp.plot_event_study(es, figs / "event_study.png",
                             f"Event study, {'+'.join(F)} L={L} m={M}, direction={pp['direction_mode']} "
                             "(bands ±1 s.e.; overlapping events)")

    # 2 heatmaps (development)
    dev_tab = ranked[PARAMS + ["sharpe"]]
    f2a = rp.plot_heatmaps(dev_tab, "H", "k", {"lookback": L, "m": M, "direction_mode": pp["direction_mode"]},
                           figs / "heatmap_k_H.png", highlight={**pp, "factors": "+".join(F)})
    f2b = rp.plot_heatmaps(dev_tab, "k", "lookback", {"m": M, "H": pp["H"], "direction_mode": pp["direction_mode"]},
                           figs / "heatmap_L_k.png", highlight={**pp, "factors": "+".join(F)})

    # 3 gamma over time
    f3 = rp.plot_gamma({f"{'+'.join(F)} L={L} m={m}": fcache.forecast(F, L, m) for m in cfg["grid"]["m"]},
                       periods, figs / "gamma.png")

    # 4 rolling betas
    fs = full_sample_ols(returns, F)
    betas = {f"L={x}": rc.betas(F, x) for x in sorted({L, 60, 250})}
    f4 = rp.plot_rolling_betas(betas, figs / "rolling_betas.png",
                               full_sample={f"beta_{f}": fs[f"Beta_{f}"] for f in F},
                               title=f"Stage-1 rolling betas, {'+'.join(F)}")

    # 5 equity + drawdown
    full = dev | val
    nets = g.net_returns(cost, rows=finalists, mask=full)
    labels = {c: ("PRIMARY " if c == finalists[0] else "alt ") + c for c in finalists}
    bh = pd.Series(np.nan_to_num(g.exret[full]), index=g.dates[full])
    f5 = rp.plot_equity({labels[c]: nets[c] for c in finalists}, bh, periods, figs / "equity.png")

    # 6 monthly entries
    cm = closes(full)
    months = g.dates[cm].to_period("M")
    ent = {}
    for c in finalists:
        s = pd.Series(g.new_trade[g.configs.index.get_loc(c), cm], index=months).groupby(level=0).agg(["sum", "size"])
        ent[labels[c]] = s.loc[s["size"] >= MIN_DAYS_PER_MONTH, "sum"]
    f6 = rp.plot_monthly_entries(ent, periods, figs / "monthly_entries.png")

    # 7 yearly table
    yearly = pd.concat({**{labels[c]: nets[c] for c in finalists}, "buy & hold GDX": bh}, axis=1)
    ytab = yearly.groupby(yearly.index.year).sum()
    ytab.index.name = "year"
    ytab.loc["dev ann. Sharpe"] = [yearly.loc[g.dates[dev], c].mean() / yearly.loc[g.dates[dev], c].std() * np.sqrt(252)
                                   for c in yearly]
    ytab.loc["val ann. Sharpe"] = [yearly.loc[g.dates[val], c].mean() / yearly.loc[g.dates[val], c].std() * np.sqrt(252)
                                   for c in yearly]
    ytab.to_csv(RESULTS / "finalists_yearly.csv")

    # 8 top 20 by neighbourhood score
    top20 = ranked.sort_values("nbhd_median", ascending=False).head(20)
    top20 = _fmt_table(top20[["nbhd_median", "nbhd_min", "nbhd_frac_positive", "passes", "n_failed"] + SECTION8])
    top20.to_csv(RESULTS / "top20_neighbourhood.csv")

    # finalists at several cost levels
    cost_rows = []
    for c in finalists:
        for name, m in (("development", dev), ("validation", val)):
            n0 = g.net_returns(0.0, rows=[c], mask=m).iloc[:, 0]
            row = {"config": labels[c], "period": name}
            for x in cfg["report"]["cost_bps"]:
                n = g.net_returns(x, rows=[c], mask=m).iloc[:, 0]
                row[f"Sharpe @ {x:g} bps"] = n.mean() / n.std() * np.sqrt(252)
                row[f"ann. return @ {x:g} bps"] = n.mean() * 252
            cost_rows.append(row)
    costs = pd.DataFrame(cost_rows).set_index(["config", "period"])

    # --- honest-reporting checks -------------------------------------------------
    # (a) closest configs: development configs failing exactly one hard filter
    one = ranked[ranked["n_failed"] == 1].copy()
    one["failed_filter"] = one[FILTERS].idxmin(axis=1)
    closest_counts = one["failed_filter"].value_counts().rename("n_configs_failing_only_this").to_frame()
    closest = _fmt_table(one.sort_values("nbhd_median", ascending=False).head(15)[
        PARAMS + ["failed_filter", "nbhd_median", "sharpe", "pct_months_with_entry", "n_entries",
                  "sharpe_first_half", "sharpe_second_half", "pct_years_positive", "ann_return_ex_top1pct_days"]])
    closest.to_csv(RESULTS / "dev_closest_configs.csv")
    vfail = pd.read_csv(RESULTS / "validation_shortlist.csv", index_col="config_id")
    vfail["failed_filters"] = vfail[FILTERS].apply(lambda r: ", ".join(f for f in FILTERS if not r[f]) or "none",
                                                   axis=1)
    vclosest = vfail[["sharpe", "n_failed", "failed_filters", "ann_return_ex_top1pct_days",
                      "sharpe_first_half", "sharpe_second_half", "pct_years_positive", "pct_months_with_entry"]]

    # (b) day-1 concentration: edge (return in trade direction) by horizon
    ev = pd.concat([d.assign(period=n) for n, d in es.items()])
    ev = ev[(ev.side == "edge") & ev.h.isin([1, 2, 3, 5, 10])]
    day1 = (ev.assign(edge_bps=ev["mean"] * 1e4)
              .pivot_table(index=["period", "k"], columns="h", values="edge_bps"))
    day1.columns = [f"cum edge to day {h} (bps)" for h in day1.columns]
    day1.to_csv(RESULTS / "event_study_edge_by_horizon.csv")
    kk = pp["k"]
    d_dev = day1.loc[("development", kk)]
    d_val = day1.loc[("validation", kk)]
    day1_text = (f"At the primary's k={kk:g}: cumulative edge after day 1 is {d_dev.iloc[0]:+.1f} bps (dev) / "
                 f"{d_val.iloc[0]:+.1f} bps (val); after day 2 {d_dev.iloc[1]:+.1f} / {d_val.iloc[1]:+.1f} bps; "
                 f"after day 10 {d_dev.iloc[-1]:+.1f} / {d_val.iloc[-1]:+.1f} bps. ")
    peak_h = int(ev[(ev.k == kk) & (ev.period == "development")].set_index("h")["mean"].idxmax())
    day1_text += ("The day-1 concentration hypothesis is <b>supported</b>." if peak_h == 1 and d_dev.iloc[0] > 0
                  else f"The day-1 concentration hypothesis is <b>not supported</b>: the edge is small on day 1 "
                       f"and peaks around day {peak_h} in development, then decays.")

    # (c) with / without 2008 and 2020
    excl = rp.exclude_years_table(nets, [2008, 2020])
    excl.index = [labels[c] for c in excl.index]
    excl.to_csv(RESULTS / "finalists_ex_years.csv")

    # (d) gross vs net viability (primary)
    pc_dev = costs.loc[(labels[finalists[0]], "development")]
    pc_val = costs.loc[(labels[finalists[0]], "validation")]
    viable = pc_val["Sharpe @ 5 bps"] > 0 and pc_dev["Sharpe @ 5 bps"] > 0
    cost_text = (f"Primary Sharpe gross / 2 bps / 5 bps: development {pc_dev['Sharpe @ 0 bps']:.2f} / "
                 f"{pc_dev['Sharpe @ 2 bps']:.2f} / {pc_dev['Sharpe @ 5 bps']:.2f}; validation "
                 f"{pc_val['Sharpe @ 0 bps']:.2f} / {pc_val['Sharpe @ 2 bps']:.2f} / {pc_val['Sharpe @ 5 bps']:.2f}. "
                 + ("The edge survives 5 bps but each bp of cost removes a visible share of it."
                    if viable else "<b>The edge does not survive 5 bps; not viable at realistic costs.</b>"))

    # text + tables
    wf = pd.read_csv(RESULTS / "walk_forward.csv", index_col="year")
    wfs = json.loads((RESULTS / "walk_forward_summary.json").read_text())
    vshort = pd.read_csv(RESULTS / "validation_shortlist.csv", index_col="config_id")
    dshort = pd.read_csv(RESULTS / "dev_shortlist.csv", index_col="config_id")
    funnel = pd.read_csv(RESULTS / "dev_filter_funnel.csv")
    vlog = pd.DataFrame(ValidationGate(splits["validation_log"]).entries())[
        ["timestamp", "evaluation_number", "n_configs", "shortlist_hash", "deviation", "reason"]]
    ok = frozen["status"] == "ok"
    banner = (f'<div class="{"ok" if ok else "warn"}"><b>Status: {html_escape(frozen["status"])}.</b> '
              + ("The primary passed every hard filter on development and validation."
                 if ok else
                 "No shortlisted config passed the hard filters on validation. The primary below is the "
                 "best by fewest failures and composite rank, and does <b>not</b> meet the stated "
                 "requirements; treat OOS results for it as a test, not a deployment.") + "</div>")
    pv, pdv = prim["validation"], prim["development"]
    ck = prim["development_checks"]
    summary = f"""{banner}
<p><b>Primary:</b> <code>{prim['config_id']}</code> &nbsp; alternates: {', '.join('<code>'+a['config_id']+'</code>' for a in frozen['alternates'])}</p>
<ul>
<li>Development net Sharpe {pdv['sharpe']:.2f} (neighbourhood median {pdv['nbhd_median']:.2f}); validation net Sharpe {pv['sharpe']:.2f}.
 Validation failed filters: {', '.join(pv['failed_filters']) or 'none'}.</li>
<li>Deflated Sharpe ratio (N = {frozen['n_configs_tested']:,} configs): {ck['dsr']:.3f}; placebo percentile {ck['placebo_percentile']:.1f};
 sign-flip Sharpe {ck['sign_flip_sharpe']:.2f}; bootstrap 95% CI [{ck['boot_sharpe_lo95']:.2f}, {ck['boot_sharpe_hi95']:.2f}].</li>
<li>Walk-forward of the selection rule 2011–2021: stitched net Sharpe {wfs['stitched_sharpe']:.2f},
 {wfs['stability']['n_distinct_configs']} distinct configs in {wfs['stability']['n_years']} years
 (red flag: {wfs['stability']['red_flag']}).</li>
<li>Frozen {frozen['frozen_at']} at commit <code>{frozen['git_commit'][:10]}</code> (dirty: {frozen['git_dirty']});
 data {frozen['data']['file']} sha256 <code>{frozen['data']['sha256'][:12]}…</code>.</li>
</ul>
<p class="muted">All returns are daily GDX excess returns on 1x notional (self-financing overlay), net of
{cost:g} bps per unit traded unless stated. Signals are formed and executed at the close of day t.
Development {periods['development'][0].date()} → {periods['development'][1].date()},
validation {periods['validation'][0].date()} → {periods['validation'][1].date()}.</p>"""

    sections = [
        ("Summary", summary),
        ("1. Event study / decay curve", rp.img_tag(f1) + "<p class='muted'>Average cumulative GDX excess "
         "return after signal closes with |z| ≥ k, by side, and in the trade's direction (right). Bands are "
         "±1 s.e. treating events as independent; overlapping events make them too narrow.</p>"),
        ("2. Development net Sharpe heatmaps", rp.img_tag(f2a) + rp.img_tag(f2b)),
        ("3. Stage-2 slope γ̂ over time", rp.img_tag(f3)),
        ("4. Stage-1 rolling betas", rp.img_tag(f4)),
        ("5. Equity curve and drawdown", rp.img_tag(f5)),
        ("6. Monthly entry counts", rp.img_tag(f6)),
        ("7. Yearly net returns (finalists)", rp.table_html(ytab)),
        ("Finalists at 0 / 2 / 5 bps (gross vs net)", f"<p>{cost_text}</p>" + rp.table_html(costs)),
        ("Does the result depend on 2008 or 2020?", rp.table_html(excl) +
         "<p class='muted'>Finalists over development + validation, net of 2 bps.</p>"),
        ("Is the edge concentrated on day 1?", f"<p>{day1_text}</p>" + rp.table_html(day1, 1)),
        ("Closest configs and the filter they fail",
         "<p>No filter was loosened. Development configs that fail exactly one hard filter "
         "(counts, then the 15 best by neighbourhood score):</p>" + rp.table_html(closest_counts)
         + rp.table_html(closest) + "<p>Shortlist on validation, with the filters each one fails:</p>"
         + rp.table_html(vclosest)),
        ("8. Top 20 configs by neighbourhood score (development)", rp.table_html(top20)),
        ("Selection funnel (development)", rp.table_html(funnel.set_index("after_filter"))),
        ("Development shortlist with statistical checks", rp.table_html(dshort)),
        ("Validation results (shortlist)", rp.table_html(vshort)),
        ("Walk-forward of the selection rule", rp.table_html(wf) + f"<pre>{html_escape(json.dumps(wfs, indent=1))}</pre>"),
        ("Validation log", rp.table_html(vlog)),
    ]
    out = rp.build_html("GDX residual-reversal strategy — in-sample report", sections, RESULTS / "report.html")
    log.info("Report written: %s", out)
    return out


def html_escape(s) -> str:
    import html as _h
    return _h.escape(str(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", default=str(ROOT / "data" / "raw" / "gold_etf_factor_regression.xlsx"))
    ap.add_argument("--validate", action="store_true", help="evaluate shortlist on validation and freeze")
    ap.add_argument("--report", action="store_true", help="rebuild report from existing frozen_config.json")
    ap.add_argument("--reason", default=None, help="required (and logged) if validation is re-run on a "
                                                   "different shortlist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(0)

    cfg = load_grid(ROOT / "config" / "grid.yaml")
    splits = load_splits(ROOT / "config" / "splits.yaml")
    workbook = Path(args.workbook)
    prices, returns = load_dataset(workbook, rf_mode=cfg["fixed"]["rf_mode"], check_adjusted=False)
    check_adjusted_closes(prices)
    RESULTS.mkdir(exist_ok=True)

    g, ranked, short = development(args, cfg, splits, returns, workbook)
    if args.validate:
        frozen = validation_and_freeze(args, cfg, splits, returns, workbook, g, ranked, short)
        make_report(cfg, splits, returns, g, ranked, _jsonable(frozen))
    elif args.report:
        frozen = json.loads((RESULTS / "frozen_config.json").read_text())
        make_report(cfg, splits, returns, g, ranked, frozen)
    else:
        log.info("Stopped before validation. Review results/dev_shortlist.csv, then rerun with --validate.")


if __name__ == "__main__":
    main()
