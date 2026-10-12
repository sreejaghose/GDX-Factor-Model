"""Dynamic factor model: lowest rolling-BIC subset each day."""
import numpy as np
import pandas as pd
import pytest

from src.factor_model import DYNAMIC, ResidualCache, factor_subsets, rolling_best_bic, rolling_residuals


def _returns(n=600, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n, name="Date")
    F = rng.normal(0, 0.01, (n, 5))
    cols = ["SPY", "IAU", "GLD", "TLT", "GDXJ"]
    df = pd.DataFrame(F, index=idx, columns=[f"ExRet_{c}" for c in cols])
    df["ExRet_GDX"] = 1.6 * df["ExRet_GLD"] + rng.normal(0, 0.01, n)   # truth: GLD only
    return df


def test_picks_lowest_bic_and_uses_its_residual():
    r = _returns()
    out, names = rolling_best_bic(r, 60)
    assert len(names) == 31
    t = 300
    best, best_bic = None, np.inf
    for s in factor_subsets(["SPY", "IAU", "GLD", "TLT", "GDXJ"]):
        f = rolling_residuals(r, s, 60).iloc[t]
        p, n = len(s) + 1, f["n_obs"]
        bic = n * np.log(f["sigma"] ** 2 * (n - p) / n) + p * np.log(n)
        if bic < best_bic:
            best, best_bic, e = "+".join(s), bic, f["e"]
    assert names[int(out["chosen"].iloc[t])] == best
    assert out["e"].iloc[t] == pytest.approx(e)
    chosen = pd.Series(np.array(names, object)[out["chosen"].dropna().astype(int)])
    assert chosen.value_counts().idxmax() == "GLD"                       # BIC finds the true model most days
    assert {"beta_SPY", "beta_GDXJ"} <= set(out.columns)


def test_candidates_need_full_window():
    r = _returns()
    r.iloc[:200, r.columns.get_loc("ExRet_GDXJ")] = np.nan   # GDXJ not yet trading
    out, names = rolling_best_bic(r, 60)
    early = out["chosen"].iloc[60:200].dropna().astype(int)
    assert not any("GDXJ" in names[i] for i in early)
    assert out["e"].iloc[60:200].notna().all()                # still live without GDXJ


def test_cache_and_no_lookahead():
    r = _returns()
    c = ResidualCache(r)
    z = c.z((DYNAMIC,), 60, 2)
    assert c.chosen_sets(60).notna().sum() > 400
    T = 350
    shocked = r.copy()
    shocked.iloc[T + 1:] = np.random.default_rng(1).normal(0, 0.05, shocked.iloc[T + 1:].shape)
    z2 = ResidualCache(shocked).z((DYNAMIC,), 60, 2)
    pd.testing.assert_series_equal(z.iloc[:T + 1], z2.iloc[:T + 1])
