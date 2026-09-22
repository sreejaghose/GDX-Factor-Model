"""Stage 1: factor regressions of GDX excess returns on factor excess returns."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["SPY", "IAU", "GLD", "TLT"]


def factor_subsets(factors: list[str] = FACTORS) -> list[tuple[str, ...]]:
    """All non-empty subsets, in the canonical SPY, IAU, GLD, TLT order."""
    return [c for n in range(1, len(factors) + 1) for c in itertools.combinations(factors, n)]


def full_sample_ols(returns: pd.DataFrame, factors: tuple[str, ...] | list[str],
                    target: str = "GDX") -> dict:
    """OLS of ExRet_<target> on ExRet_<factors> with intercept (same as Excel LINEST).

    Returns alpha (bps/day), betas, t-stats, two-sided p-values, R2, adj R2,
    AIC and BIC. AIC/BIC use the Gaussian log-likelihood with k+1 mean
    parameters, the same as the workbook: N ln(2pi) + N ln(SSR/N) + N + pen*(k+1).
    """
    cols = [f"ExRet_{f}" for f in factors]
    df = returns[[f"ExRet_{target}", *cols]].dropna()
    y = df[f"ExRet_{target}"]
    X = sm.add_constant(df[cols])
    res = sm.OLS(y, X).fit()
    out = {
        "Factors": "+".join(factors),
        "NumFactors": len(factors),
        "N": int(res.nobs),
        "Alpha_bps": res.params["const"] * 1e4,
        "Alpha_t": res.tvalues["const"],
        "Alpha_p": res.pvalues["const"],
        "R2": res.rsquared,
        "AdjR2": res.rsquared_adj,
        "AIC": res.aic,
        "BIC": res.bic,
    }
    for f, c in zip(factors, cols):
        out[f"Beta_{f}"] = res.params[c]
        out[f"t_{f}"] = res.tvalues[c]
        out[f"p_{f}"] = res.pvalues[c]
    return out


def full_sample_sweep(returns: pd.DataFrame, target: str = "GDX") -> pd.DataFrame:
    """``full_sample_ols`` for every factor subset, one row per subset."""
    return pd.DataFrame([full_sample_ols(returns, s, target) for s in factor_subsets()])
