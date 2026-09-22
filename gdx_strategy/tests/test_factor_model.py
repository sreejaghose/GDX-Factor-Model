"""Rolling Stage-1 fit vs a brute-force per-day OLS."""
import numpy as np
import pandas as pd
import pytest

from src.factor_model import ResidualCache, check_factor_set, residual_score, rolling_residuals


def synthetic_returns(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    F = rng.normal(0, 0.01, (n, 4))
    y = 0.0002 + F @ np.array([0.5, 0.0, 1.7, 0.1]) + rng.normal(0, 0.015, n)
    df = pd.DataFrame(F, index=idx, columns=[f"ExRet_{c}" for c in ("SPY", "IAU", "GLD", "TLT")])
    df["ExRet_GDX"] = y
    return df


def brute_force(df, factors, L):
    y = df["ExRet_GDX"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[[f"ExRet_{f}" for f in factors]].to_numpy()])
    p = X.shape[1]
    e = np.full(len(df), np.nan)
    sig = np.full(len(df), np.nan)
    betas = np.full((len(df), p), np.nan)
    for t in range(L, len(df)):
        Xw, yw = X[t - L:t], y[t - L:t]
        b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        r = yw - Xw @ b
        betas[t] = b
        sig[t] = np.sqrt(r @ r / (L - p))
        e[t] = y[t] - X[t] @ b
    return betas, e, sig


@pytest.mark.parametrize("factors,L", [(("SPY", "GLD"), 60), (("SPY", "GLD", "TLT"), 120), (("GLD",), 30)])
def test_matches_brute_force(factors, L):
    df = synthetic_returns()
    r = rolling_residuals(df, factors, L)
    betas, e, sig = brute_force(df, factors, L)
    assert r.iloc[:L].drop(columns="n_obs").isna().all().all()
    np.testing.assert_allclose(r["e"].to_numpy()[L:], e[L:], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(r["sigma"].to_numpy()[L:], sig[L:], rtol=1e-8)
    coef = r[["alpha", *[f"beta_{f}" for f in factors]]].to_numpy()
    np.testing.assert_allclose(coef[L:], betas[L:], rtol=1e-7, atol=1e-10)


def test_score_definition():
    df = synthetic_returns()
    r = rolling_residuals(df, ("SPY", "GLD"), 60)
    z3 = residual_score(r, 3)
    t = 100
    expected = r["e"].iloc[t - 2:t + 1].sum() / (r["sigma"].iloc[t] * np.sqrt(3))
    assert z3.iloc[t] == pytest.approx(expected)
    assert z3.iloc[:62].isna().all() and z3.iloc[62:].notna().all()
    np.testing.assert_allclose(residual_score(r, 1), r["e"] / r["sigma"])


def test_missing_data_is_skipped_not_propagated():
    df = synthetic_returns()
    df.iloc[150, df.columns.get_loc("ExRet_SPY")] = np.nan
    r = rolling_residuals(df, ("SPY", "GLD"), 60, min_obs_frac=0.9)
    assert np.isnan(r["e"].iloc[150])        # no residual on the missing day
    assert r["n_obs"].iloc[160] == 59        # window still fits with one fewer obs
    assert np.isfinite(r["e"].iloc[160])


def test_rejects_iau_with_gld():
    with pytest.raises(ValueError):
        check_factor_set(("SPY", "IAU", "GLD"))
    assert check_factor_set(("GLD", "SPY")) == ("SPY", "GLD")


def test_cache_returns_same_object():
    c = ResidualCache(synthetic_returns())
    assert c.z(("GLD", "SPY"), 60, 2) is c.z(("SPY", "GLD"), 60, 2)
    assert list(c.betas(("SPY", "GLD"), 60).columns) == ["alpha", "beta_SPY", "beta_GLD"]
