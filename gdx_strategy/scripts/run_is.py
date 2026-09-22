"""Full in-sample workflow.

  python scripts/run_is.py              # development only: grid, filters, ranking,
                                        # shortlist + statistical checks (no validation)
  python scripts/run_is.py --validate   # ... then evaluate the shortlist on validation
                                        # (logged), pick primary + alternates, freeze,
                                        # and run the anchored walk-forward check

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
from src.metrics import grid_metrics  # noqa: E402
from src.robustness import (FILTERS, filter_funnel, hard_filters, make_selection_rule,  # noqa: E402
                            rank_configs, simplicity_key, statistical_checks)
from src.splits import evaluate_validation, load_splits, split_masks, walk_forward  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", default=str(ROOT / "data" / "raw" / "gold_etf_factor_regression.xlsx"))
    ap.add_argument("--validate", action="store_true", help="evaluate shortlist on validation and freeze")
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
        validation_and_freeze(args, cfg, splits, returns, workbook, g, ranked, short)
    else:
        log.info("Stopped before validation. Review results/dev_shortlist.csv, then rerun with --validate.")


if __name__ == "__main__":
    main()
