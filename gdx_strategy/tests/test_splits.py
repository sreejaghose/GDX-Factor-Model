"""IS protocol: split masks, dev-only == full-run slice, validation gate, walk-forward."""
import numpy as np
import pandas as pd
import pytest

from src.data import load_dataset
from src.grid import GridResult, load_grid, run_grid
from src.splits import (ValidationGate, evaluate_validation, load_splits, split_masks,
                        walk_forward)

from conftest import ROOT


def best_sharpe(grid, mask):
    """Placeholder rule for tests only (the real rule is defined in robustness)."""
    net = grid.net_returns(0.0, mask=mask)
    return (net.mean() / net.std()).idxmax()


# ---- real data ----------------------------------------------------------------

@pytest.fixture(scope="module")
def real(is_workbook):
    cfg = load_grid(ROOT / "config" / "grid.yaml")
    cfg["grid"].update(factors=[("SPY", "GLD")], lookback=[60, 250], m=[1, 2], k=[1.0, 1.5], H=[1, 5])
    _, r = load_dataset(is_workbook, rf_mode=cfg["fixed"]["rf_mode"], check_adjusted=False)
    return cfg, r, run_grid(r, cfg)


def test_masks_partition_scored_rows(real):
    _, _, g = real
    s = load_splits(ROOT / "config" / "splits.yaml")
    m = split_masks(g, s)
    assert not (m["development"] & m["validation"]).any()
    assert ((m["development"] | m["validation"]) == g.eval_mask).all()
    d = g.dates
    assert d[m["development"]][0] > g.signal_start and d[m["development"]][-1] <= pd.Timestamp("2016-12-31")
    assert d[m["validation"]][0] >= pd.Timestamp("2017-01-01") and d[m["validation"]][-1] == d[-1]


def test_dev_only_run_equals_full_run_slice(real):
    """Development results cannot depend on validation data."""
    cfg, r, g = real
    s = load_splits(ROOT / "config" / "splits.yaml")
    g_dev = run_grid(r.loc[:s["development"]["end"]], cfg)
    assert g_dev.signal_start == g.signal_start
    n = len(g_dev.dates)
    np.testing.assert_array_equal(g_dev.pos, g.pos[:, :n])
    dev = split_masks(g, s)["development"]
    pd.testing.assert_frame_equal(g_dev.net_returns(2.0, mask=dev[:n]), g.net_returns(2.0, mask=dev))


# ---- validation gate ------------------------------------------------------------

def test_validation_gate(tmp_path):
    gate = ValidationGate(tmp_path / "log.jsonl")
    e1 = gate.check_and_log(["a", "b"])
    assert e1["evaluation_number"] == 1 and not e1["deviation"]
    assert not gate.check_and_log(["b", "a"])["deviation"]  # same shortlist, any order
    with pytest.raises(PermissionError):
        gate.check_and_log(["a", "c"])
    e3 = gate.check_and_log(["a", "c"], reason="bug fix in engine")
    assert e3["deviation"] and e3["reason"] == "bug fix in engine"
    assert len(gate.entries()) == 3


def test_evaluate_validation_uses_gate(real, tmp_path):
    _, _, g = real
    s = load_splits(ROOT / "config" / "splits.yaml")
    s["validation_log"] = tmp_path / "val.jsonl"
    ids = list(g.configs.index[:2])
    v = evaluate_validation(g, ids, s, cost_bps=2.0)
    assert list(v.columns) == ids and v.index[0] >= pd.Timestamp("2017-01-01")
    with pytest.raises(PermissionError):
        evaluate_validation(g, list(g.configs.index[2:4]), s, cost_bps=2.0)


# ---- walk-forward on synthetic grid ------------------------------------------------

def synthetic_grid(seed=0):
    """3 configs, always long / short / flat-ish; GDX drifts up 2008-2013, down after."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", "2016-12-31")
    drift = np.where(dates.year <= 2013, 1e-3, -1e-3)
    r = drift + rng.normal(0, 0.01, len(dates))
    pos = np.vstack([np.ones(len(dates)), -np.ones(len(dates)),
                     (rng.random(len(dates)) < 0.1).astype(float)]).astype(np.int8)
    pos[:, :5] = 0
    new = np.zeros_like(pos)
    new[:, 5] = 1
    configs = pd.DataFrame(dict(factors=[("GLD",)] * 3, lookback=[20, 40, 60], m=1, k=1.0, H=5,
                                policy="A_run_out", direction_mode=["estimated", "reversion", "estimated"]),
                           index=pd.Index(["long", "short", "flat"], name="config_id"))
    return GridResult(configs, dates, r, pos, new, dates[4], 3)


def test_walk_forward_choices_and_switch_cost():
    g = synthetic_grid()
    wf = walk_forward(g, best_sharpe, 2010, 2016, cost_bps=10)
    t = wf["years"]
    assert list(t.index) == list(range(2010, 2017))
    assert (t.loc[2010:2014, "config_id"] == "long").all()      # trained on up-years
    assert t.loc[2016, "config_id"] == "short" or t.loc[2015, "config_id"] == "long"
    sw = t["config_id"] != t["config_id"].shift()
    sw.iloc[0] = False
    # always-long -> always-short: 2 units traded, new config's own first-row cost was 0
    assert (t.loc[sw, "switch_cost"] == pytest.approx(2 * 10e-4)).all()
    assert (t.loc[~sw, "switch_cost"] == 0).all()
    assert len(wf["returns"]) == sum((g.dates.year == y).sum() for y in range(2010, 2017))
    assert wf["stability"]["n_years"] == 7


def test_walk_forward_uses_only_past():
    g = synthetic_grid()
    base = walk_forward(g, best_sharpe, 2010, 2016, cost_bps=2)["years"]["config_id"]
    g2 = synthetic_grid()
    later = g2.dates.year >= 2014
    g2.exret[later] = -g2.exret[later] * 3  # rewrite the future
    alt = walk_forward(g2, best_sharpe, 2010, 2016, cost_bps=2)["years"]["config_id"]
    assert (base.loc[2010:2014] == alt.loc[2010:2014]).all()
