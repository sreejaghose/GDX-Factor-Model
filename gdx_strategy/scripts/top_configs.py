"""Top configs by development net Sharpe after a chosen set of hard filters.

  python scripts/top_configs.py                       # default filters, top 25, both all-factors runs
  python scripts/top_configs.py --top 50 --runs results/variants/all_factors_2008

Filters (applied on development, net of 2 bps):
  * net Sharpe > 0 in both halves of development
  * >= 70% of calendar years positive
  * at least one new entry in 100% of calendar months
  * policy != D_stack_2x (reference only)
Validation columns are shown for information; they play no part in the ranking.
Writes results/top_configs/<run>_top<N>.csv and prints the tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [ROOT / "results" / "variants" / "all_factors_2008", ROOT / "results" / "variants" / "all_factors_common"]
OUT = ROOT / "results" / "top_configs"
DEV_COLS = ["sharpe", "sharpe_gross", "ann_return", "max_drawdown", "nw_tstat", "n_entries", "pct_months_with_entry",
            "pct_years_positive", "sharpe_first_half", "sharpe_second_half", "ann_return_ex_top1pct_days", "hit_rate"]
VAL_COLS = ["sharpe", "ann_return", "max_drawdown", "pct_months_with_entry", "pct_years_positive",
            "sharpe_first_half", "sharpe_second_half"]


def top_configs(run_dir: Path, top: int, min_years: float = 0.70, min_months: float = 1.0) -> tuple[pd.DataFrame, dict]:
    s = pd.read_csv(run_dir / "all_configs_summary.csv.gz")
    s = s[s["policy"] != "D_stack_2x"]
    params = ["factors", "lookback", "m", "k", "H", "policy", "direction_mode"]
    dev = s[s.period == "development"].set_index("config_id")
    val = s[s.period == "validation"].set_index("config_id")
    keep = ((dev.sharpe_first_half > 0) & (dev.sharpe_second_half > 0)
            & (dev.pct_years_positive >= min_years - 1e-12) & (dev.pct_months_with_entry >= min_months - 1e-12))
    funnel = {"configs": len(dev), "both halves > 0": int(((dev.sharpe_first_half > 0) & (dev.sharpe_second_half > 0)).sum()),
              "+ years >= 70%": int(((dev.sharpe_first_half > 0) & (dev.sharpe_second_half > 0)
                                     & (dev.pct_years_positive >= min_years - 1e-12)).sum()),
              "+ every month": int(keep.sum())}
    d = dev[keep].sort_values("sharpe", ascending=False).head(top)
    out = d[params].copy()
    out.insert(0, "rank", range(1, len(d) + 1))
    for c in DEV_COLS:
        out[f"dev_{c}"] = d[c]
    for c in VAL_COLS:
        out[f"val_{c}"] = val.loc[d.index, c]
    nb_path = run_dir / "grid" / "dev_ranked_all.csv.gz"
    if nb_path.exists():
        nb = pd.read_csv(nb_path, index_col="config_id", usecols=["config_id", "nbhd_median"])
        out["dev_nbhd_median"] = nb.loc[d.index, "nbhd_median"]
    return out, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--runs", nargs="*", default=[str(p) for p in DEFAULT_RUNS])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    for r in map(Path, args.runs):
        t, funnel = top_configs(r, args.top)
        t.to_csv(OUT / f"{r.name}_top{args.top}.csv", index=False)
        print(f"\n=== {r.name}: {funnel}")
        show = ["rank", "factors", "lookback", "m", "k", "H", "policy", "direction_mode", "dev_sharpe", "dev_nbhd_median",
                "dev_n_entries", "val_sharpe", "val_pct_months_with_entry", "val_sharpe_first_half", "val_sharpe_second_half"]
        print(t[[c for c in show if c in t]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
