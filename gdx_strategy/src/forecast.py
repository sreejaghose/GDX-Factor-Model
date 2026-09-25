"""Stage 2: one-day-ahead predictive regression of GDX excess return on z.

Row t uses the pairs (z_s, ExRet_GDX_{s+1}) for s in [t-W, t-1] (or all s <= t-1
when expanding). The newest target is ExRet_GDX_t, which is known at the close
of t. Forecast r_hat_{t+1} = gamma_t * z_t is therefore also known at that close.

Main fit has no intercept so GDX's negative drift doesn't become a permanent
short bias; the with-intercept fit is kept as a diagnostic.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.factor_model import ResidualCache, check_factor_set

log = logging.getLogger(__name__)

DirectionMode = Literal["estimated", "reversion"]
DEFAULT_MIN_OBS = 250


def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def predictive_regression(z: pd.Series, exret: pd.Series, window: int | None = None,
                          min_obs: int = DEFAULT_MIN_OBS) -> pd.DataFrame:
    """Expanding (window=None) or rolling-W fit of ExRet_{s+1} on z_s, evaluated on day t.

    Columns:
      z, gamma, gamma_t, n_pairs, r_hat           (no-intercept model)
      ic_alpha, ic_alpha_t, ic_gamma, ic_gamma_t  (with-intercept diagnostic)
    Rows with fewer than ``min_obs`` valid pairs, or with z_t missing, get NaN
    in the forecast.
    """
    z = z.astype(float)
    exret = exret.reindex(z.index).astype(float)
    x = z.to_numpy()
    y = exret.shift(-1).to_numpy()  # pair on row s: (z_s, ExRet_{s+1})
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = np.where(ok, x, 0.0), np.where(ok, y, 0.0)

    P = {k: _prefix(v) for k, v in dict(n=ok.astype(float), x=xv, y=yv, xx=xv * xv,
                                        xy=xv * yv, yy=yv * yv).items()}
    N = len(x)
    t = np.arange(N)
    lo = np.zeros(N, dtype=int) if window is None else np.maximum(t - int(window), 0)
    S = {k: v[t] - v[lo] for k, v in P.items()}  # sums over pair rows [lo, t-1]
    n = S["n"]

    with np.errstate(invalid="ignore", divide="ignore"):
        # no intercept
        gamma = S["xy"] / S["xx"]
        ssr = np.maximum(S["yy"] - gamma * S["xy"], 0.0)
        se = np.sqrt(ssr / (n - 1) / S["xx"])
        gamma_t = gamma / se
        # with intercept
        sxx = S["xx"] - S["x"] ** 2 / n
        sxy = S["xy"] - S["x"] * S["y"] / n
        syy = S["yy"] - S["y"] ** 2 / n
        b = sxy / sxx
        a = (S["y"] - b * S["x"]) / n
        s2 = np.maximum(syy - b * sxy, 0.0) / (n - 2)
        b_t = b / np.sqrt(s2 / sxx)
        a_t = a / np.sqrt(s2 * (1.0 / n + (S["x"] / n) ** 2 / sxx))

    fit = n >= min_obs
    out = pd.DataFrame({
        "z": x,
        "gamma": np.where(fit, gamma, np.nan),
        "gamma_t": np.where(fit, gamma_t, np.nan),
        "n_pairs": n,
        "ic_alpha": np.where(fit, a, np.nan),
        "ic_alpha_t": np.where(fit, a_t, np.nan),
        "ic_gamma": np.where(fit, b, np.nan),
        "ic_gamma_t": np.where(fit, b_t, np.nan),
    }, index=z.index)
    out["r_hat"] = out["gamma"] * out["z"]
    return out


def trade_direction(fc: pd.DataFrame, mode: DirectionMode = "estimated") -> pd.Series:
    """+1 long / -1 short / 0 flat or unavailable. The |z| >= k gate is applied later.

    estimated : sign(r_hat_{t+1})  -- data decides reversion vs continuation
    reversion : -sign(z_t)         -- imposes mean reversion (only needs z, but is
                                      gated on r_hat being available so both modes
                                      start on the same day)
    """
    if mode == "estimated":
        d = np.sign(fc["r_hat"])
    elif mode == "reversion":
        d = -np.sign(fc["z"]).where(fc["r_hat"].notna())
    else:
        raise ValueError(f"direction_mode must be 'estimated' or 'reversion', got {mode!r}")
    return d.fillna(0.0).astype(int).rename("direction")


def summarize(fc: pd.DataFrame) -> dict:
    """Diagnostic snapshot, logged: final gamma (both models) and share of days gamma < 0."""
    g = fc["gamma"].dropna()
    last = fc.dropna(subset=["gamma"]).iloc[-1] if len(g) else None
    s = {
        "first_forecast": g.index[0].date() if len(g) else None,
        "n_forecasts": int(len(g)),
        "share_gamma_negative": float((g < 0).mean()) if len(g) else np.nan,
        "final_gamma_bps": float(last["gamma"] * 1e4) if last is not None else np.nan,
        "final_gamma_t": float(last["gamma_t"]) if last is not None else np.nan,
        "final_ic_alpha_bps": float(last["ic_alpha"] * 1e4) if last is not None else np.nan,
        "final_ic_gamma_bps": float(last["ic_gamma"] * 1e4) if last is not None else np.nan,
        "final_ic_gamma_t": float(last["ic_gamma_t"]) if last is not None else np.nan,
    }
    log.info("Stage 2: gamma=%.2f bps (t=%.2f); intercept model alpha=%.2f bps, gamma=%.2f bps "
             "(t=%.2f); gamma<0 on %.0f%% of days", s["final_gamma_bps"], s["final_gamma_t"],
             s["final_ic_alpha_bps"], s["final_ic_gamma_bps"], s["final_ic_gamma_t"],
             100 * s["share_gamma_negative"])
    return s


class ForecastCache:
    """Memoises Stage-2 output per (F, L, m) on top of a ResidualCache (W is fixed)."""

    def __init__(self, resid_cache: ResidualCache, window: int | None = None,
                 min_obs: int = DEFAULT_MIN_OBS):
        self.rc = resid_cache
        self.window = window
        self.min_obs = min_obs
        self._fc: dict[tuple, pd.DataFrame] = {}

    def forecast(self, factors, lookback: int, m: int) -> pd.DataFrame:
        key = (check_factor_set(factors), int(lookback), int(m))
        if key not in self._fc:
            z = self.rc.z(*key)
            exret = self.rc.returns[f"ExRet_{self.rc.target}"]
            self._fc[key] = predictive_regression(z, exret, self.window, self.min_obs)
        return self._fc[key]
