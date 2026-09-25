"""Grid enumeration, common evaluation window, and consistency with backtest()."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest
from src.data import load_dataset
from src.factor_model import ResidualCache
from src.forecast import ForecastCache
from src.grid import GridResult, enumerate_configs, load_grid, n_configs, run_grid

from conftest import ROOT

GRID_YAML = ROOT / "config" / "grid.yaml"


def test_grid_yaml_count():
    cfg = load_grid(GRID_YAML)
    assert n_configs(cfg["grid"]) == 10_800
    c = enumerate_configs(cfg["grid"])
    assert len(c) == 10_800 and c.index.is_unique
    assert not any(("IAU" in f and "GLD" in f) for f in c["factors"])


@pytest.fixture(scope="module")
def small(is_workbook):
    cfg = load_grid(GRID_YAML)
    cfg["grid"].update(factors=[("SPY", "GLD")], lookback=[20, 250], m=[1, 5], k=[1.0, 2.0], H=[1, 5])
    _, r = load_dataset(is_workbook, rf_mode=cfg["fixed"]["rf_mode"], check_adjusted=False)
    return cfg, r, run_grid(r, cfg, include_reference=True)


def test_common_start_is_slowest_config(small):
    cfg, r, g = small
    fx = cfg["fixed"]
    fcache = ForecastCache(ResidualCache(r, min_obs_frac=fx["stage1_min_obs_frac"]),
                           window=fx["stage2_window"], min_obs=fx["stage2_min_obs"])
    slow = fcache.forecast(("SPY", "GLD"), 250, 5)["r_hat"].first_valid_index()
    fast = fcache.forecast(("SPY", "GLD"), 20, 1)["r_hat"].first_valid_index()
    assert g.signal_start == slow > fast
    assert (g.pos[:, np.asarray(g.dates < g.signal_start)] == 0).all()
    assert g.eval_mask.sum() == (g.dates > slow).sum()
    assert g.n_tested == 1 * 2 * 2 * 2 * 2 * 3 * 2
    assert len(g.configs) == g.n_tested // 3 * 4  # + D reference


@pytest.mark.parametrize("i", [0, 7, 29, 63])
def test_net_matches_backtest(small, i):
    cfg, r, g = small
    c = g.configs.iloc[i]
    fx = cfg["fixed"]
    fcache = ForecastCache(ResidualCache(r, min_obs_frac=fx["stage1_min_obs_frac"]),
                           window=fx["stage2_window"], min_obs=fx["stage2_min_obs"])
    fc = fcache.forecast(c["factors"], c["lookback"], c["m"]).copy()
    fc.loc[fc.index < g.signal_start, ["r_hat", "z"]] = np.nan  # same common start
    res = backtest(fc, r["ExRet_GDX"], c["k"], c["H"], c["policy"], c["direction_mode"], cost_bps=2)
    net = g.net_returns(2.0, rows=[c.name]).iloc[:, 0]
    np.testing.assert_allclose(net.to_numpy(), res.daily["net"].loc[net.index].to_numpy(), atol=1e-7)
    assert g.new_trade[i].sum() == len(res.trades)


def test_save_load_roundtrip(small, tmp_path):
    _, _, g = small
    g.save(tmp_path / "grid.npz")
    h = GridResult.load(tmp_path / "grid.npz")
    assert h.signal_start == g.signal_start and h.n_tested == g.n_tested
    np.testing.assert_array_equal(h.pos, g.pos)
    pd.testing.assert_frame_equal(h.net_returns(2.0), g.net_returns(2.0))
