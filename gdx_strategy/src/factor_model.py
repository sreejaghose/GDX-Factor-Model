"""Stage 1: factor regressions of GDX excess returns on factor excess returns.

Full-sample OLS (workbook reproduction) and the rolling out-of-window residual
score used by the strategy.

Rolling timing convention (row t = information at the close of day t):
  * betas on row t are fitted on trading days [t-L, t-1] -- day t excluded;
  * e_t = ExRet_GDX_t - alpha_t - beta_t' ExRet_F_t uses day t's returns, so it
    is known at the close of t and never enters its own fit.
"""
from __future__ import annotations

import itertools
import math

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


# ---------------------------------------------------------------------------
# Rolling out-of-window residuals
# ---------------------------------------------------------------------------

def check_factor_set(factors) -> tuple[str, ...]:
    """Validate and canonicalise a factor set (never IAU and GLD together)."""
    factors = tuple(factors)
    unknown = set(factors) - set(FACTORS)
    if unknown or not factors or len(set(factors)) != len(factors):
        raise ValueError(f"Invalid factor set {factors!r}")
    if "IAU" in factors and "GLD" in factors:
        raise ValueError("IAU and GLD are near-duplicates; never use both in one model")
    return tuple(f for f in FACTORS if f in factors)


def rolling_residuals(returns: pd.DataFrame, factors, lookback: int, target: str = "GDX",
                      min_obs_frac: float = 0.9) -> pd.DataFrame:
    """Rolling OLS on the window [t-L, t-1], evaluated out of window on day t.

    Uses prefix sums of X'X, X'y and y'y so each window is a subtraction and
    all p x p solves are batched: O(N p^2), no Python loop over days.

    Columns:
      alpha, beta_<F>  coefficients from the [t-L, t-1] fit
      e                out-of-window residual on day t
      sigma            in-window residual std, sqrt(SSR / (n - p))
      n_obs            valid observations in the window
    Rows without a valid fit (warm-up, too few obs, missing data on t) are NaN.
    """
    factors = check_factor_set(factors)
    L = int(lookback)
    p = len(factors) + 1
    if L <= p + 1:
        raise ValueError(f"lookback {L} too short for {p} parameters")
    min_obs = max(p + 2, math.ceil(min_obs_frac * L))

    y = returns[f"ExRet_{target}"].to_numpy(float)
    F = returns[[f"ExRet_{f}" for f in factors]].to_numpy(float)
    N = len(y)
    X = np.column_stack([np.ones(N), F])
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xv = np.where(valid[:, None], X, 0.0)
    yv = np.where(valid, y, 0.0)

    def prefix(a):  # prefix[i] = sum of rows [0, i)
        return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])

    C_xx = prefix(Xv[:, :, None] * Xv[:, None, :])
    C_xy = prefix(Xv * yv[:, None])
    C_yy = prefix(yv * yv)
    C_n = prefix(valid.astype(float))

    t = np.arange(L, N)  # window rows [t-L, t-1] -> prefix[t] - prefix[t-L]
    XtX = C_xx[t] - C_xx[t - L]
    Xty = C_xy[t] - C_xy[t - L]
    yty = C_yy[t] - C_yy[t - L]
    n = C_n[t] - C_n[t - L]

    ok = (n >= min_obs) & valid[t]
    XtX_safe = np.where(ok[:, None, None], XtX, np.eye(p))
    beta = np.linalg.solve(XtX_safe, np.where(ok[:, None], Xty, 0.0)[:, :, None])[:, :, 0]
    ssr = np.maximum(yty - np.einsum("ij,ij->i", beta, Xty), 0.0)
    sigma = np.sqrt(ssr / np.maximum(n - p, 1))
    e = y[t] - np.einsum("ij,ij->i", X[t], beta)

    out = np.full((N, p + 3), np.nan)
    out[t] = np.column_stack([beta, e, sigma, n])
    out[t[~ok], :] = np.nan
    out[t[~ok], -1] = n[~ok]
    cols = ["alpha", *[f"beta_{f}" for f in factors], "e", "sigma", "n_obs"]
    return pd.DataFrame(out, index=returns.index, columns=cols)


def residual_score(resid: pd.DataFrame, m: int) -> pd.Series:
    """z_t = S_t / (sigma_t * sqrt(m)),  S_t = sum_{j=0}^{m-1} e_{t-j}.

    Each e is from its own day's fit; sigma is from day t's fit. Needs all m
    residuals to be present.
    """
    m = int(m)
    if m < 1:
        raise ValueError("m must be >= 1")
    S = resid["e"].rolling(m, min_periods=m).sum()
    z = S / (resid["sigma"] * np.sqrt(m))
    z.name = "z"
    return z


class ResidualCache:
    """Memoises rolling fits per (F, L) and scores per (F, L, m) for one dataset."""

    def __init__(self, returns: pd.DataFrame, target: str = "GDX", min_obs_frac: float = 0.9):
        self.returns = returns
        self.target = target
        self.min_obs_frac = min_obs_frac
        self._resid: dict[tuple, pd.DataFrame] = {}
        self._z: dict[tuple, pd.Series] = {}

    def residuals(self, factors, lookback: int) -> pd.DataFrame:
        key = (check_factor_set(factors), int(lookback))
        if key not in self._resid:
            self._resid[key] = rolling_residuals(self.returns, key[0], key[1],
                                                 self.target, self.min_obs_frac)
        return self._resid[key]

    def z(self, factors, lookback: int, m: int) -> pd.Series:
        key = (check_factor_set(factors), int(lookback), int(m))
        if key not in self._z:
            self._z[key] = residual_score(self.residuals(key[0], key[1]), key[2])
        return self._z[key]

    def betas(self, factors, lookback: int) -> pd.DataFrame:
        r = self.residuals(factors, lookback)
        return r[[c for c in r.columns if c == "alpha" or c.startswith("beta_")]]
