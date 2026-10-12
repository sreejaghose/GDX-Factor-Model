"""Single-regression (factor-only) variant: no Stage 2, direction = -sign(z)."""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.backtest import backtest, build_signal
from src.data import load_dataset
from src.factor_model import ResidualCache
from src.forecast import ForecastCache, forecast_cache, signal_live, trade_direction
from src.grid import enumerate_configs, load_grid, n_configs, run_configs, run_grid

VARIANT = ROOT / "config" / "grid_factor_only.yaml"


def _returns(n=700, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n, name="Date")
    return pd.DataFrame(rng.normal(0, 0.01, (n, 5)), index=idx,
                        columns=[f"ExRet_{c}" for c in ("SPY", "IAU", "GLD", "TLT", "GDX")])


def test_factor_only_has_no_stage2_and_is_live_with_z():
    r = _returns()
    rc = ResidualCache(r)
    fo = ForecastCache(rc, model="factor_only").forecast(("SPY", "GLD"), 60, 2)
    two = ForecastCache(rc, min_obs=250).forecast(("SPY", "GLD"), 60, 2)
    pd.testing.assert_series_equal(fo["z"], two["z"])           # same single regression
    assert fo["gamma"].isna().all() and fo["r_hat"].isna().all()
    assert (signal_live(fo) == fo["z"].notna()).all()
    assert signal_live(fo).idxmax() < signal_live(two).idxmax()  # no 250-day Stage-2 warm-up
    d = trade_direction(fo, "reversion")
    live = fo["z"].notna() & (fo["z"] != 0)
    assert (d[live] == -np.sign(fo.loc[live, "z"])).all()
    assert (d[~fo["z"].notna()] == 0).all()


def test_factor_only_rejects_stage2_features():
    fo = ForecastCache(ResidualCache(_returns()), model="factor_only").forecast(("GLD",), 40, 1)
    with pytest.raises(ValueError):
        trade_direction(fo, "estimated")
    with pytest.raises(ValueError):
        build_signal(fo, 1.0, "reversion", min_edge_bps=2.0)
    with pytest.raises(ValueError):
        ForecastCache(ResidualCache(_returns()), model="three_stage")


def test_same_positions_as_two_stage_reversion_once_both_live():
    """With one shared start after Stage 2 is live, reversion trades are identical."""
    r = _returns()
    base = dict(stage1_min_obs_frac=0.9, stage2_window=None, stage2_min_obs=250, min_edge_bps=0.0)
    cfgs = enumerate_configs(dict(factors=[("SPY", "GLD")], lookback=[40], m=[1, 3], k=[0.75],
                                  H=[2, 5], policy=["A_run_out", "B_reset_flip"], direction_mode=["reversion"]))
    start = r.index[400]
    g2 = run_configs(r, cfgs, {**base, "model": "two_stage"}, start)
    g1 = run_configs(r, cfgs, {**base, "model": "factor_only"}, start)
    np.testing.assert_array_equal(g1.pos, g2.pos)


def test_backtest_scores_from_first_live_z():
    r = _returns()
    fo = ForecastCache(ResidualCache(r), model="factor_only").forecast(("GLD",), 20, 1)
    res = backtest(fo, r["ExRet_GDX"], k=1.0, H=2, policy="B_reset_flip", direction_mode="reversion")
    first_live = signal_live(fo).idxmax()
    assert res.daily["net"].first_valid_index() == res.daily.index[res.daily.index > first_live][0]


def test_variant_config_and_pinned_window(is_workbook):
    cfg = load_grid(VARIANT)
    assert cfg["fixed"]["model"] == "factor_only" and cfg["grid"]["direction_mode"] == ["reversion"]
    assert n_configs(cfg["grid"]) == 5_400 and cfg["fixed"]["prior_trials"] == 10_800
    cfg["grid"].update(lookback=[20, 250], m=[1, 5], k=[1.0], H=[2])
    _, r = load_dataset(is_workbook, rf_mode=cfg["fixed"]["rf_mode"], check_adjusted=False)
    g = run_grid(r, cfg)
    assert g.signal_start == pd.Timestamp("2008-05-23")   # same evaluation window as two-stage
    assert (g.pos[:, np.asarray(g.dates < g.signal_start)] == 0).all()


def test_factor_only_no_lookahead(is_workbook):
    from src.data import build_returns, load_fedfunds, load_prices
    fixed = load_grid(VARIANT)["fixed"]
    px, ff = load_prices(is_workbook), load_fedfunds(is_workbook)
    T = pd.Timestamp("2015-03-31")
    shocked = px.copy()
    after = px.index > T
    shocked.loc[after] = px.loc[after].to_numpy() * np.exp(
        np.random.default_rng(7).normal(0, 0.1, shocked.loc[after].shape))
    ra, rb = build_returns(px, ff, fixed["rf_mode"]), build_returns(shocked, ff, fixed["rf_mode"])
    cfgs = enumerate_configs(dict(factors=[("GLD",), ("SPY", "GLD", "TLT")], lookback=[20], m=[1, 2], k=[0.75],
                                  H=[2], policy=["A_run_out"], direction_mode=["reversion"]))
    ga = run_configs(ra, cfgs, fixed, pd.Timestamp("2008-05-23"))
    gb = run_configs(rb, cfgs, fixed, pd.Timestamp("2008-05-23"))
    upto = np.asarray(ga.dates <= T)
    np.testing.assert_array_equal(ga.pos[:, upto], gb.pos[:, upto])
    za = forecast_cache(ra, fixed).forecast(("GLD",), 20, 1)["z"]
    zb = forecast_cache(rb, fixed).forecast(("GLD",), 20, 1)["z"]
    pd.testing.assert_series_equal(za.loc[:T], zb.loc[:T])
