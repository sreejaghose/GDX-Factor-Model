"""Anything dated t must be unchanged when data after t is altered."""
import numpy as np
import pandas as pd

from src.factor_model import residual_score, rolling_residuals


def _returns(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 5)), index=idx,
                      columns=[f"ExRet_{c}" for c in ("SPY", "IAU", "GLD", "TLT", "GDX")])
    return df


def test_stage1_uses_no_future_data():
    df = _returns()
    t = 200
    shocked = df.copy()
    shocked.iloc[t + 1:] = np.random.default_rng(99).normal(0, 0.05, shocked.iloc[t + 1:].shape)
    for m in (1, 3):
        a = residual_score(rolling_residuals(df, ("SPY", "GLD", "TLT"), 60), m)
        b = residual_score(rolling_residuals(shocked, ("SPY", "GLD", "TLT"), 60), m)
        np.testing.assert_array_equal(a.iloc[:t + 1].to_numpy(), b.iloc[:t + 1].to_numpy())


def test_day_t_excluded_from_its_own_fit():
    """Changing only day t moves e_t, but not the betas/sigma used on day t."""
    df = _returns()
    t = 200
    bumped = df.copy()
    bumped.iloc[t, bumped.columns.get_loc("ExRet_GDX")] += 0.05
    a = rolling_residuals(df, ("SPY", "GLD"), 60)
    b = rolling_residuals(bumped, ("SPY", "GLD"), 60)
    cols = ["alpha", "beta_SPY", "beta_GLD", "sigma"]
    np.testing.assert_array_equal(a[cols].iloc[t].to_numpy(), b[cols].iloc[t].to_numpy())
    assert abs(b["e"].iloc[t] - a["e"].iloc[t] - 0.05) < 1e-12
    assert not np.allclose(a[cols].iloc[t + 1].to_numpy(), b[cols].iloc[t + 1].to_numpy())


def test_stage2_forecast_uses_no_future_data():
    from src.forecast import predictive_regression

    df = _returns(n=600)
    z = residual_score(rolling_residuals(df, ("SPY", "GLD"), 60), 2)
    r = df["ExRet_GDX"]
    t = 450
    base = predictive_regression(z, r, min_obs=250)

    # Future returns (t+1 onward), and future z, must not affect anything dated <= t
    r_f = r.copy()
    r_f.iloc[t + 1:] *= -5
    z_f = z.copy()
    z_f.iloc[t + 1:] = 9.0
    fut = predictive_regression(z_f, r_f, min_obs=250)
    cols = ["gamma", "gamma_t", "r_hat", "ic_gamma"]
    np.testing.assert_array_equal(base[cols].iloc[:t + 1].to_numpy(), fut[cols].iloc[:t + 1].to_numpy())

    # ... but day t's own return (known at t's close) IS the newest target used on day t
    r_t = r.copy()
    r_t.iloc[t] += 0.05
    now = predictive_regression(z, r_t, min_obs=250)
    assert now["gamma"].iloc[t] != base["gamma"].iloc[t]
    np.testing.assert_array_equal(now["gamma"].iloc[:t].to_numpy(), base["gamma"].iloc[:t].to_numpy())
