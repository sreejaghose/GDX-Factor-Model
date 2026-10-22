"""Compare every factor combination for GDX: regression fit and strategy results.

  python scripts/factor_combinations.py

Reads the two exploration runs produced by run_is.py:
  results/variants/all_factors_2008    (15 sets of SPY/IAU/GLD/TLT, original 2008 window)
  results/variants/all_factors_common  (31 sets incl. GDXJ, one common late window)
and writes results/factor_combinations/ (CSV tables, charts, report.html).

Regression sweep: full-sample OLS of GDX excess return on each of the 31 subsets of
SPY, IAU, GLD, TLT, GDXJ, (a) on each subset's own sample and (b) on the common
sample where GDXJ exists, the only basis on which AIC/BIC are comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import report as rp  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.factor_model import FACTORS, collinearity, full_sample_sweep  # noqa: E402
from src.robustness import FILTERS  # noqa: E402

OUT = ROOT / "results" / "factor_combinations"
RUNS = {"2008 window (no GDXJ)": ROOT / "results" / "variants" / "all_factors_2008",
        "common window (incl. GDXJ)": ROOT / "results" / "variants" / "all_factors_common"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1d1d1f", "#6b6b66", "#e4e4e0"
ORIGINAL = {"GLD", "SPY+GLD", "SPY+GLD+TLT"}


def regression_sweeps(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for name, common in (("own_sample", False), ("common_sample", True)):
        t = full_sample_sweep(returns, "GDX", FACTORS, common_sample=common)
        sample = returns.dropna(subset=[f"ExRet_{c}" for c in ["GDX", *FACTORS]]) if common else returns
        coll = pd.DataFrame([collinearity(sample, f.split("+")) for f in t["Factors"]])
        t = pd.concat([t, coll], axis=1)
        t["has_GDXJ"] = t["Factors"].str.contains("GDXJ")
        t["IAU_and_GLD"] = t["Factors"].str.contains("IAU") & t["Factors"].str.contains("GLD")
        if common:
            t["BIC_rank"] = t["BIC"].rank().astype(int)
            t["AIC_rank"] = t["AIC"].rank().astype(int)
        cols = (["Factors", "NumFactors", "N", "Alpha_bps", "Alpha_t", "R2", "AdjR2", "AIC", "BIC"]
                + [c for c in t.columns if c.startswith(("Beta_", "t_"))]
                + ["cond_number", "max_vif", "has_GDXJ", "IAU_and_GLD"] + (["BIC_rank", "AIC_rank"] if common else []))
        out[name] = t[cols].sort_values("BIC" if common else "R2", ascending=common)
    return out


def strategy_summary(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Per factor set: Sharpe distribution in development and validation, how many
    configs pass the development hard filters, and how well dev rank carries over."""
    s = pd.read_csv(run_dir / "all_configs_summary.csv.gz")
    s = s[s["policy"] != "D_stack_2x"]
    dev = s[s.period == "development"].set_index("config_id")
    val = s[s.period == "validation"].set_index("config_id")
    ranked_p = run_dir / "grid" / "dev_ranked_all.csv.gz"
    ranked = pd.read_csv(ranked_p, index_col="config_id") if ranked_p.exists() else None
    rows = []
    for fs, d in dev.groupby("factors"):
        v = val.loc[d.index]
        top = d["sharpe"].rank(pct=True) >= 0.9
        row = {"factors": fs, "n_factors": fs.count("+") + 1, "configs": len(d),
               "dev median Sharpe": d["sharpe"].median(), "dev 90th pct Sharpe": d["sharpe"].quantile(0.9),
               "dev best Sharpe": d["sharpe"].max(), "dev % > 0": (d["sharpe"] > 0).mean(),
               "val median Sharpe": v["sharpe"].median(), "val % > 0": (v["sharpe"] > 0).mean(),
               "val mean Sharpe of dev top 10%": v.loc[top, "sharpe"].mean(),
               "dev→val rank corr": d["sharpe"].corr(v["sharpe"], method="spearman")}
        if ranked is not None:
            r = ranked.loc[d.index]
            row["pass dev filters"] = int(r["passes"].sum())
            row["best nbhd median"] = r["nbhd_median"].max()
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("factors").sort_values("dev median Sharpe", ascending=False)
    frozen = json.loads((run_dir / "frozen_config.json").read_text())
    return tab, frozen


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_regression(common: pd.DataFrame, path: Path) -> Path:
    t = common.sort_values("BIC", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True, gridspec_kw={"wspace": 0.05})
    y = np.arange(len(t))
    a1.barh(y, t["R2"], color=BLUE, height=0.7)
    a2.barh(y, t["BIC"] - t["BIC"].min(), color=ORANGE, height=0.7)
    a1.set_yticks(y, t["Factors"])
    for ax, lab in ((a1, "R² (common sample)"), (a2, "BIC − best BIC (lower is better)")):
        _style(ax)
        ax.set_xlabel(lab, color=MUTED)
    for i, (r2, gx) in enumerate(zip(t["R2"], t["has_GDXJ"])):
        a1.text(r2 + 0.005, i, f"{r2:.3f}", va="center", fontsize=7, color=INK)
    a1.set_xlim(0, 1.0)
    fig.suptitle(f"GDX factor regressions, all 31 combinations, common sample "
                 f"(N = {int(common['N'].iloc[0]):,}, from GDXJ inception)", x=0.01, y=0.95, ha="left", color=INK)
    fig.subplots_adjust(top=0.92)
    return rp._save(fig, path)


def plot_strategy(tab: pd.DataFrame, title: str, path: Path) -> Path:
    t = tab.sort_values("dev median Sharpe")
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.hlines(y, t[["dev median Sharpe", "val median Sharpe"]].min(axis=1),
              t[["dev median Sharpe", "val median Sharpe"]].max(axis=1), color=GRID, lw=2, zorder=1)
    ax.scatter(t["dev median Sharpe"], y, color=BLUE, s=42, zorder=3, label="development median",
               edgecolor="white", linewidth=1.5)
    ax.scatter(t["val median Sharpe"], y, color=ORANGE, s=42, zorder=3, label="validation median",
               edgecolor="white", linewidth=1.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    labels = [f"{f}  ★" if f in ORIGINAL else f for f in t.index]
    ax.set_yticks(y, labels)
    _style(ax)
    ax.set_xlabel("median net Sharpe across that factor set's 3,600 configs (2 bps)", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return rp._save(fig, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, returns = load_dataset(ROOT / "data" / "raw" / "gold_etf_factor_regression.xlsx", rf_mode="prior_month",
                              check_adjusted=False)
    sweeps = regression_sweeps(returns)
    for name, t in sweeps.items():
        t.to_csv(OUT / f"regression_sweep_{name}.csv", index=False)
    f_reg = plot_regression(sweeps["common_sample"], OUT / "regression_common_sample.png")

    sections = []
    common = sweeps["common_sample"]
    best = common.iloc[0]
    reg_text = (f"<p>Common sample: {common['N'].iloc[0]:,} days from GDXJ's inception (2009-11-12) to 2021-12-31. "
                f"Best BIC: <b>{best['Factors']}</b> (R² {best['R2']:.3f}). Every model that includes GDXJ explains "
                f"far more of GDX (R² ≈ {common[common.has_GDXJ]['R2'].min():.2f}–{common[common.has_GDXJ]['R2'].max():.2f}) "
                f"than any without it (≤ {common[~common.has_GDXJ]['R2'].max():.2f}): GDXJ is itself a basket of gold "
                "miners, so with it in the model the residual becomes GDX-versus-junior-miners relative value rather "
                "than GDX-versus-gold. IAU and GLD together give a maximum VIF of "
                f"{common[common.IAU_and_GLD]['max_vif'].max():,.0f}: near-perfectly collinear, the split between their "
                "two betas is noise even though the fit barely changes.</p>")
    show = ["Factors", "N", "R2", "AdjR2", "BIC", "BIC_rank", "Alpha_bps", "Beta_SPY", "Beta_IAU", "Beta_GLD",
            "Beta_TLT", "Beta_GDXJ", "t_GDXJ", "max_vif"]
    sections.append(("1. Regression fit, common sample (AIC/BIC comparable)",
                     reg_text + rp.img_tag(f_reg) + rp.table_html(common[show].set_index("Factors"), 3)))
    own = sweeps["own_sample"]
    sections.append(("Regression fit, each set on its own sample (N differs; compare R² only)",
                     rp.table_html(own[["Factors", "N", "R2", "AdjR2", "Alpha_bps", "max_vif"]].set_index("Factors"), 3)))

    frozen_rows = []
    for i, (label, run_dir) in enumerate(RUNS.items(), start=2):
        if not (run_dir / "frozen_config.json").exists():
            sections.append((f"{i}. Strategy grid — {label}", "<p>Not run yet.</p>"))
            continue
        tab, frozen = strategy_summary(run_dir)
        tab.to_csv(OUT / f"strategy_by_factor_set_{run_dir.name}.csv")
        f = plot_strategy(tab, f"Strategy results by factor set — {label} (★ = original grid's sets)",
                          OUT / f"strategy_{run_dir.name}.png")
        p, d, v = frozen["primary"], frozen["primary"]["development"], frozen["primary"]["validation"]
        frozen_rows.append({"run": label, "dev window": " → ".join(frozen["splits"]["development"]),
                            "configs": frozen["n_configs_tested"], "DSR trials": frozen.get("n_trials_for_dsr"),
                            "primary": p["config_id"], "dev Sharpe": d["sharpe"], "val Sharpe": v["sharpe"],
                            "val failed filters": ", ".join(v["failed_filters"]) or "none",
                            "DSR": p["development_checks"]["dsr"], "status": frozen["status"]})
        sections.append((f"{i}. Strategy grid — {label}",
                         f"<p>Each factor set has 3,600 configs (L × m × k × H × policy × direction). Full report: "
                         f"<code>{run_dir.relative_to(ROOT)}/report.html</code>.</p>" + rp.img_tag(f)
                         + rp.table_html(tab, 3)))
    if frozen_rows:
        sections.insert(1, ("Selected configuration per run (same selection rule as the original)",
                            rp.table_html(pd.DataFrame(frozen_rows).set_index("run"), 3)))
    rp.build_html("GDX — all factor combinations", sections, OUT / "report.html")
    print(f"Wrote {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
