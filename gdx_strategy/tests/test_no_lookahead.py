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


def test_positions_and_pnl_use_no_future_data():
    from src.backtest import backtest
    from src.forecast import predictive_regression

    df = _returns(n=700)

    def run(d):
        z = residual_score(rolling_residuals(d, ("SPY", "GLD"), 60), 2)
        fc = predictive_regression(z, d["ExRet_GDX"], min_obs=250)
        return backtest(fc, d["ExRet_GDX"], k=1.0, H=3, policy="B_reset_flip").daily

    t = 500
    shocked = df.copy()
    shocked.iloc[t + 1:] = np.random.default_rng(5).normal(0, 0.05, shocked.iloc[t + 1:].shape)
    a, b = run(df), run(shocked)
    assert (a["pos"].iloc[:t + 1] != 0).any()
    np.testing.assert_array_equal(a["pos"].iloc[:t + 1], b["pos"].iloc[:t + 1])
    np.testing.assert_array_equal(a["net"].iloc[:t + 1], b["net"].iloc[:t + 1])  # P&L through t


# ---------------------------------------------------------------------------
# Look-ahead / correctness checklist on the real workbook
# ---------------------------------------------------------------------------

import json  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import ROOT  # noqa: E402


def _checklist_configs():
    from src.grid import config_id
    rows = [dict(factors=("GLD",), lookback=20, m=1, k=0.75, H=2, policy="A_run_out", direction_mode="reversion"),
            dict(factors=("SPY", "GLD"), lookback=60, m=2, k=1.0, H=5, policy="B_reset_flip",
                 direction_mode="estimated"),
            dict(factors=("SPY", "GLD", "TLT"), lookback=250, m=5, k=0.5, H=10, policy="C_reset_flat",
                 direction_mode="reversion")]
    df = pd.DataFrame(rows)
    df.index = [config_id(r) for r in rows]
    return df


def test_future_price_perturbation_real_data(is_workbook):
    """Randomise every price after T: z_t, r_hat_{t+1} and pos_t for t <= T are unchanged."""
    from src.data import build_returns, load_fedfunds, load_prices
    from src.forecast import ForecastCache
    from src.factor_model import ResidualCache
    from src.grid import load_grid, run_configs

    fixed = load_grid(ROOT / "config" / "grid.yaml")["fixed"]
    px, ff = load_prices(is_workbook), load_fedfunds(is_workbook)
    T = pd.Timestamp("2014-06-30")
    after = px.index > T
    shocked = px.copy()
    rng = np.random.default_rng(123)
    shocked.loc[after] = px.loc[after].to_numpy() * np.exp(rng.normal(0, 0.1, shocked.loc[after].shape))
    ra, rb = build_returns(px, ff, fixed["rf_mode"]), build_returns(shocked, ff, fixed["rf_mode"])
    assert not np.allclose(ra.loc[ra.index > T, "ExRet_GDX"], rb.loc[rb.index > T, "ExRet_GDX"])
    cfgs = _checklist_configs()
    start = pd.Timestamp("2008-05-23")
    ga, gb = run_configs(ra, cfgs, fixed, start), run_configs(rb, cfgs, fixed, start)
    upto = np.asarray(ga.dates <= T)
    np.testing.assert_array_equal(ga.pos[:, upto], gb.pos[:, upto])
    assert (ga.pos[:, upto] != 0).any(axis=1).all()
    fa = ForecastCache(ResidualCache(ra, min_obs_frac=fixed["stage1_min_obs_frac"]), min_obs=fixed["stage2_min_obs"])
    fb = ForecastCache(ResidualCache(rb, min_obs_frac=fixed["stage1_min_obs_frac"]), min_obs=fixed["stage2_min_obs"])
    for _, c in cfgs.iterrows():
        a, b = fa.forecast(c["factors"], c["lookback"], c["m"]), fb.forecast(c["factors"], c["lookback"], c["m"])
        pd.testing.assert_frame_equal(a.loc[:T, ["z", "gamma", "r_hat"]], b.loc[:T, ["z", "gamma", "r_hat"]])
        assert not a.loc[a.index > T, "r_hat"].equals(b.loc[b.index > T, "r_hat"])


def test_oracle_pnl_uses_next_day_return():
    """pos_t earns r_{t+1}: an oracle on r_{t+1} is absurdly good; one on r_t is not."""
    from src.backtest import backtest

    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2010-01-01", periods=2500)
    r = pd.Series(rng.normal(0, 0.02, len(idx)), idx)

    def sharpe_for(direction):
        # reversion mode trades -sign(z); feed z = -direction so the trade is `direction`
        fc = pd.DataFrame({"z": -direction * 3.0, "r_hat": 1e-4}, index=idx)
        res = backtest(fc, r, k=1.0, H=1, policy="B_reset_flip", direction_mode="reversion", cost_bps=0)
        n = res.net
        return n.mean() / n.std() * np.sqrt(252)

    cheat = np.sign(r.shift(-1)).fillna(0.0)   # knows tomorrow's return
    honest = np.sign(r).fillna(0.0)            # knows only today's return
    assert sharpe_for(cheat) > 20
    assert abs(sharpe_for(honest)) < 1.5


def test_strategy_uses_prior_month_rf(is_workbook):
    from src.data import daily_rf, load_fedfunds
    from src.grid import load_grid

    assert load_grid(ROOT / "config" / "grid.yaml")["fixed"]["rf_mode"] == "prior_month"
    frozen = json.loads((ROOT / "results" / "frozen_config.json").read_text())
    assert frozen["fixed"]["rf_mode"] == "prior_month"
    ff = load_fedfunds(is_workbook)
    d = pd.DatetimeIndex(["2010-03-01", "2010-03-31", "2019-08-15"])
    rf = daily_rf(d, ff, "prior_month")
    assert rf.iloc[0] == rf.iloc[1] == ff.loc["2010-02-01"] / 100
    assert rf.iloc[2] == ff.loc["2019-07-01"] / 100


def test_all_configs_share_one_evaluation_start(is_workbook):
    from src.data import load_dataset
    from src.grid import load_grid, run_grid

    cfg = load_grid(ROOT / "config" / "grid.yaml")
    cfg["grid"].update(lookback=[20, 250], m=[1, 5], k=[0.5, 2.0], H=[1, 10])
    _, r = load_dataset(is_workbook, rf_mode=cfg["fixed"]["rf_mode"], check_adjusted=False)
    g = run_grid(r, cfg)
    net = g.net_returns(2.0)
    assert net.index[0] == g.dates[g.dates > g.signal_start][0]      # one index for every column
    assert net.notna().all().all()
    first_trade = [g.dates[np.flatnonzero(p)[0]] for p in g.pos if p.any()]
    assert min(first_trade) >= g.signal_start


def test_every_config_result_is_persisted():
    """All configs (winners and losers) and the tested count are on disk and consistent."""
    summ = ROOT / "results" / "all_configs_summary.csv.gz"
    man = ROOT / "results" / "grid_manifest.json"
    if not summ.exists() or not man.exists():
        pytest.skip("run scripts/run_is.py --report first")
    s = pd.read_csv(summ)
    m = json.loads(man.read_text())
    frozen = json.loads((ROOT / "results" / "frozen_config.json").read_text())
    assert m["n_selectable"] == frozen["n_configs_tested"] == 10_800
    assert s["config_id"].nunique() == m["n_configs_total"] == 14_400
    assert (s.groupby("period").size() == m["n_configs_total"]).all()
    sel = s[(s.period == "development") & (s.policy != "D_stack_2x")]
    assert len(sel) == 10_800 and (sel["sharpe"] < 0).sum() > 1000   # losers are kept
    p = frozen["primary"]
    got = s[(s.config_id == p["config_id"]) & (s.period == "development")]["sharpe"].iloc[0]
    assert got == pytest.approx(p["development"]["sharpe"], abs=1e-5)
