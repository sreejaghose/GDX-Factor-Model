"""Stage-2 predictive regression vs statsmodels on explicit windows."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.forecast import predictive_regression, trade_direction


def _data(n=700, gamma=-0.002, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2011-01-03", periods=n)
    z = rng.normal(size=n)
    r = np.r_[0.0, gamma * z[:-1]] + rng.normal(0, 0.02, n) - 0.0003
    z = pd.Series(z, idx)
    z.iloc[:40] = np.nan  # Stage-1 warm-up
    return z, pd.Series(r, idx)


@pytest.mark.parametrize("window", [None, 300])
@pytest.mark.parametrize("t", [400, 699])
def test_matches_statsmodels(window, t):
    z, r = _data()
    fc = predictive_regression(z, r, window=window, min_obs=250)
    s_lo = 0 if window is None else t - window
    x = z.iloc[s_lo:t].to_numpy()          # s in [t-W, t-1]
    y = r.iloc[s_lo + 1:t + 1].to_numpy()  # ExRet_{s+1}; newest is ExRet_t
    ok = np.isfinite(x)
    ni = sm.OLS(y[ok], x[ok]).fit()
    ic = sm.OLS(y[ok], sm.add_constant(x[ok])).fit()
    row = fc.iloc[t]
    assert row["n_pairs"] == ok.sum()
    assert row["gamma"] == pytest.approx(ni.params[0], rel=1e-9)
    assert row["gamma_t"] == pytest.approx(ni.tvalues[0], rel=1e-7)
    assert row["ic_alpha"] == pytest.approx(ic.params[0], rel=1e-7, abs=1e-12)
    assert row["ic_gamma"] == pytest.approx(ic.params[1], rel=1e-7)
    assert row["ic_gamma_t"] == pytest.approx(ic.tvalues[1], rel=1e-6)
    assert row["ic_alpha_t"] == pytest.approx(ic.tvalues[0], rel=1e-6)
    assert row["r_hat"] == pytest.approx(ni.params[0] * z.iloc[t], rel=1e-9)


def test_min_obs_gate_and_recovers_sign():
    z, r = _data()
    fc = predictive_regression(z, r, min_obs=250)
    first = fc["gamma"].first_valid_index()
    assert fc.loc[first, "n_pairs"] == 250
    assert fc["gamma"].loc[:first].iloc[:-1].isna().all()
    assert fc["gamma"].iloc[-1] < 0 and fc["gamma_t"].iloc[-1] < -2


def test_direction_modes():
    fc = pd.DataFrame({"z": [2.0, -3.0, 1.0, 2.5, np.nan],
                       "r_hat": [-1e-3, 2e-3, 1e-3, np.nan, np.nan]})
    assert trade_direction(fc, "estimated").tolist() == [-1, 1, 1, 0, 0]
    assert trade_direction(fc, "reversion").tolist() == [-1, 1, -1, 0, 0]
    with pytest.raises(ValueError):
        trade_direction(fc, "momentum")
