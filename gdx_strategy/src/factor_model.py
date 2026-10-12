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
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

WORKBOOK_FACTORS = ["SPY", "IAU", "GLD", "TLT"]   # the workbook sweep's factor pool
FACTORS = WORKBOOK_FACTORS + ["GDXJ"]             # every factor the code accepts (canonical order)
DYNAMIC = "BEST_BIC"   # pseudo factor set: each day, the lowest-BIC model among all subsets of FACTORS


def factor_subsets(factors: list[str] = WORKBOOK_FACTORS) -> list[tuple[str, ...]]:
    """All non-empty subsets of ``factors``, in canonical order."""
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


def full_sample_sweep(returns: pd.DataFrame, target: str = "GDX", factors: list[str] = WORKBOOK_FACTORS,
                      common_sample: bool = False) -> pd.DataFrame:
    """``full_sample_ols`` for every subset of ``factors``, one row per subset.

    common_sample=True fits every subset on the same rows (where the target and all
    of ``factors`` exist), which is required for AIC/BIC to be comparable across
    subsets; otherwise each subset uses all rows where its own series exist.
    """
    if common_sample:
        cols = [f"ExRet_{c}" for c in [target, *factors]]
        returns = returns.loc[returns[cols].notna().all(axis=1)]
    return pd.DataFrame([full_sample_ols(returns, s, target) for s in factor_subsets(factors)])


def collinearity(returns: pd.DataFrame, factors) -> dict:
    """Condition number of the standardised factor matrix and the largest VIF."""
    X = returns[[f"ExRet_{f}" for f in factors]].dropna()
    if X.shape[1] == 1:
        return {"cond_number": 1.0, "max_vif": 1.0}
    Z = ((X - X.mean()) / X.std()).to_numpy()
    corr_inv = np.linalg.inv(np.corrcoef(Z, rowvar=False))
    return {"cond_number": float(np.linalg.cond(Z)), "max_vif": float(np.diag(corr_inv).max())}


# ---------------------------------------------------------------------------
# Rolling out-of-window residuals
# ---------------------------------------------------------------------------

def check_factor_set(factors) -> tuple[str, ...]:
    """Validate and canonicalise a factor set.

    Whether IAU and GLD (near-duplicates) may appear together is a config-level
    rule enforced by grid.load_grid; see has_iau_and_gld.
    """
    factors = tuple(factors)
    if factors == (DYNAMIC,):
        return factors
    unknown = set(factors) - set(FACTORS)
    if unknown or not factors or len(set(factors)) != len(factors):
        raise ValueError(f"Invalid factor set {factors!r}")
    return tuple(f for f in FACTORS if f in factors)


def has_iau_and_gld(factors) -> bool:
    return "IAU" in factors and "GLD" in factors


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
    rhs = np.where(ok[:, None], Xty, 0.0)[:, :, None]
    try:
        beta = np.linalg.solve(XtX_safe, rhs)[:, :, 0]
    except np.linalg.LinAlgError:  # an exactly singular window (e.g. duplicate factors): least squares
        beta = (np.linalg.pinv(XtX_safe) @ rhs)[:, :, 0]
    ssr = np.maximum(yty - np.einsum("ij,ij->i", beta, Xty), 0.0)
    sigma = np.sqrt(ssr / np.maximum(n - p, 1))
    e = y[t] - np.einsum("ij,ij->i", X[t], beta)

    out = np.full((N, p + 3), np.nan)
    out[t] = np.column_stack([beta, e, sigma, n])
    out[t[~ok], :] = np.nan
    out[t[~ok], -1] = n[~ok]
    cols = ["alpha", *[f"beta_{f}" for f in factors], "e", "sigma", "n_obs"]
    return pd.DataFrame(out, index=returns.index, columns=cols)


def rolling_best_bic(returns: pd.DataFrame, lookback: int, pool: list[str] = FACTORS, target: str = "GDX",
                     min_obs_frac: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    """Dynamic factor model: each day, fit every non-empty subset of ``pool`` on the window
    [t-L, t-1] and use the one with the lowest BIC for day t's residual.

    BIC_t = n ln(SSR/n) + p ln(n) on the window (Gaussian likelihood; the constants are
    the same for every candidate). BIC is only comparable on identical rows, so on
    each day only candidates fitted on the full row count (the maximum n among valid
    fits) compete; e.g. GDXJ models are excluded until their window is complete.
    The choice uses only the window, so it is known at the close of t-1.

    Returns (frame, subset names). Frame columns: alpha, beta_<f> for every f in pool
    (0 when f is not in the chosen model), e, sigma, n_obs, chosen (index into the
    subset names), bic.
    """
    subsets = factor_subsets(pool)
    names = ["+".join(s) for s in subsets]
    fits = [rolling_residuals(returns, s, lookback, target, min_obs_frac) for s in subsets]
    N = len(returns)
    e = np.column_stack([f["e"].to_numpy() for f in fits])
    sig = np.column_stack([f["sigma"].to_numpy() for f in fits])
    n = np.column_stack([f["n_obs"].to_numpy() for f in fits])
    p = np.array([len(s) + 1 for s in subsets], float)
    ok = np.isfinite(e) & np.isfinite(sig)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # warm-up rows have no valid fit
        ssr = np.maximum(sig ** 2 * (n - p[None, :]), 1e-300)
        bic = n * np.log(ssr / n) + p[None, :] * np.log(n)
        nmax = np.nanmax(np.where(ok, n, np.nan), axis=1)
    eligible = ok & (n == nmax[:, None])
    bic_e = np.where(eligible, bic, np.inf)
    any_ok = eligible.any(axis=1)
    choice = np.argmin(bic_e, axis=1)
    rows = np.arange(N)
    cols = ["alpha", *[f"beta_{f}" for f in pool], "e", "sigma", "n_obs", "chosen", "bic"]
    out = pd.DataFrame(np.nan, index=returns.index, columns=cols)
    out["alpha"] = np.where(any_ok, np.column_stack([f["alpha"].to_numpy() for f in fits])[rows, choice], np.nan)
    for f in pool:
        b = np.column_stack([fit[f"beta_{f}"].to_numpy() if f"beta_{f}" in fit else np.zeros(N) for fit in fits])
        out[f"beta_{f}"] = np.where(any_ok, b[rows, choice], np.nan)
    out["e"] = np.where(any_ok, e[rows, choice], np.nan)
    out["sigma"] = np.where(any_ok, sig[rows, choice], np.nan)
    out["n_obs"] = np.where(any_ok, n[rows, choice], np.nan)
    out["chosen"] = np.where(any_ok, choice, np.nan)
    out["bic"] = np.where(any_ok, bic_e[rows, choice], np.nan)
    return out, names


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

    def __init__(self, returns: pd.DataFrame, target: str = "GDX", min_obs_frac: float = 0.9,
                 dynamic_pool: list[str] = FACTORS):
        self.returns = returns
        self.target = target
        self.min_obs_frac = min_obs_frac
        self.dynamic_pool = list(dynamic_pool)
        self.dynamic_names: list[str] = []
        self._resid: dict[tuple, pd.DataFrame] = {}
        self._z: dict[tuple, pd.Series] = {}

    def residuals(self, factors, lookback: int) -> pd.DataFrame:
        key = (check_factor_set(factors), int(lookback))
        if key not in self._resid:
            if key[0] == (DYNAMIC,):
                self._resid[key], self.dynamic_names = rolling_best_bic(
                    self.returns, key[1], self.dynamic_pool, self.target, self.min_obs_frac)
            else:
                self._resid[key] = rolling_residuals(self.returns, key[0], key[1],
                                                     self.target, self.min_obs_frac)
        return self._resid[key]

    def chosen_sets(self, lookback: int) -> pd.Series:
        """BEST_BIC only: name of the factor set chosen on each day (NaN before live)."""
        r = self.residuals((DYNAMIC,), lookback)
        names = np.array(self.dynamic_names, dtype=object)
        out = pd.Series(np.nan, index=r.index, dtype=object)
        ok = r["chosen"].notna()
        out[ok] = names[r.loc[ok, "chosen"].astype(int)]
        return out.rename("chosen_set")

    def z(self, factors, lookback: int, m: int) -> pd.Series:
        key = (check_factor_set(factors), int(lookback), int(m))
        if key not in self._z:
            self._z[key] = residual_score(self.residuals(key[0], key[1]), key[2])
        return self._z[key]

    def betas(self, factors, lookback: int) -> pd.DataFrame:
        r = self.residuals(factors, lookback)
        return r[[c for c in r.columns if c == "alpha" or c.startswith("beta_")]]
