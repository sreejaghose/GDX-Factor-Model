"""Hand-built signal sequences for every policy, timing and cost accounting."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, build_signal, run_engine, strategy_returns


def pos_of(sig, H, policy):
    return run_engine(np.array(sig), H, policy)[0].tolist()


# ---- basic holding period ---------------------------------------------------

@pytest.mark.parametrize("policy", ["A_run_out", "B_reset_flip", "C_reset_flat", "D_stack_2x"])
def test_single_signal_held_exactly_H(policy):
    # entered at close 1, closed at close 1+H=4 -> earns returns 2, 3, 4
    assert pos_of([0, 1, 0, 0, 0, 0], 3, policy) == [0, 1, 1, 1, 0, 0]
    assert pos_of([0, -1, 0, 0, 0, 0], 3, policy) == [0, -1, -1, -1, 0, 0]
    assert pos_of([0, 1, 0, 0], 1, policy) == [0, 1, 0, 0]


# ---- A_run_out ----------------------------------------------------------------

def test_A_ignores_same_and_opposite():
    assert pos_of([1, 0, 1, 0, 0, 0, 0], 3, "A_run_out") == [1, 1, 1, 0, 0, 0, 0]
    assert pos_of([1, -1, -1, 0, 0, 0], 3, "A_run_out") == [1, 1, 1, 0, 0, 0]


def test_A_signal_on_expiry_day_opens_new_trade():
    pos, tid, new, ex = run_engine(np.array([1, 0, 0, 1, 0, 0, 0]), 3, "A_run_out")
    assert pos.tolist() == [1, 1, 1, 1, 1, 1, 0]
    assert new.tolist() == [1, 0, 0, 1, 0, 0, 0]
    assert tid.tolist() == [1, 1, 1, 2, 2, 2, 0]
    assert ex.tolist() == [0, 0, 0, 1, 0, 0, 1]
    # opposite signal on expiry day: expire, then enter short
    assert pos_of([1, 0, 0, -1, 0, 0, 0], 3, "A_run_out") == [1, 1, 1, -1, -1, -1, 0]


# ---- B_reset_flip ---------------------------------------------------------------

def test_B_same_direction_resets_clock():
    assert pos_of([1, 0, 1, 0, 0, 0, 0], 3, "B_reset_flip") == [1, 1, 1, 1, 1, 0, 0]


def test_B_opposite_flips_with_new_clock():
    pos, tid, new, ex = run_engine(np.array([1, 0, -1, 0, 0, 0, 0]), 3, "B_reset_flip")
    assert pos.tolist() == [1, 1, -1, -1, -1, 0, 0]
    assert new.tolist() == [1, 0, 1, 0, 0, 0, 0]
    assert ex.tolist() == [0, 0, 2, 0, 0, 1, 0]


# ---- C_reset_flat ---------------------------------------------------------------

def test_C_same_direction_resets_clock():
    assert pos_of([-1, 0, -1, 0, 0, 0, 0], 3, "C_reset_flat") == [-1, -1, -1, -1, -1, 0, 0]


def test_C_opposite_goes_flat_without_new_trade():
    pos, tid, new, ex = run_engine(np.array([1, 0, -1, 0, 0, 0]), 3, "C_reset_flat")
    assert pos.tolist() == [1, 1, 0, 0, 0, 0]
    assert new.tolist() == [1, 0, 0, 0, 0, 0]
    assert ex.tolist() == [0, 0, 3, 0, 0, 0]
    # a later signal opens a fresh trade
    assert pos_of([1, 0, -1, -1, 0, 0, 0], 3, "C_reset_flat") == [1, 1, 0, -1, -1, -1, 0]


# ---- D_stack_2x (reference only) ----------------------------------------------

def test_D_stacks_to_2x_and_caps():
    assert pos_of([1, 1, 0, 0, 0, 0], 3, "D_stack_2x") == [1, 2, 2, 2, 0, 0]
    assert pos_of([1, 1, 1, 0, 0, 0, 0], 3, "D_stack_2x") == [1, 2, 2, 2, 2, 0, 0]
    assert pos_of([1, 1, -1, 0, 0, 0], 3, "D_stack_2x") == [1, 2, -1, -1, -1, 0]


def test_H1_consecutive_signals():
    for p in ("A_run_out", "B_reset_flip", "C_reset_flat"):
        pos, tid, new, _ = run_engine(np.array([1, 1, 1, 0]), 1, p)
        assert pos.tolist() == [1, 1, 1, 0]
        assert new.tolist() == [1, 1, 1, 0]  # each day is a fresh 1-day trade


def test_input_validation():
    with pytest.raises(ValueError):
        run_engine(np.array([2, 0]), 3, "A_run_out")
    with pytest.raises(ValueError):
        run_engine(np.array([1, 0]), 0, "A_run_out")
    with pytest.raises(ValueError):
        run_engine(np.array([1, 0]), 3, "E_unknown")


# ---- P&L timing and costs -----------------------------------------------------

def test_returns_timing_and_costs():
    pos = np.array([0, 1, 1, -1, -1, 0, 0])
    r = np.array([0.5, 0.01, 0.02, 0.03, -0.04, 0.05, 0.06])
    gross, cost, net = strategy_returns(pos, r, cost_bps=10)
    # strat_{t+1} = pos_t * r_{t+1}; row t's own return is never earned by pos_t
    np.testing.assert_allclose(gross, [0, 0, 0.02, 0.03, 0.04, -0.05, 0])
    np.testing.assert_allclose(gross, np.r_[0, pos[:-1] * r[1:]])
    # entry at close 1 (1 unit) -> booked row 2; flip at close 3 (2 units) -> row 4; exit at 5 -> row 6
    np.testing.assert_allclose(cost, [0, 0, 1e-3, 0, 2e-3, 0, 1e-3])
    np.testing.assert_allclose(net, gross - cost)


def _fc(z, r_hat):
    idx = pd.bdate_range("2020-01-01", periods=len(z))
    return pd.DataFrame({"z": z, "r_hat": r_hat}, index=idx)


def test_build_signal_gates():
    fc = _fc([np.nan, 2.5, -1.0, -3.0, 2.0, 2.2], [np.nan, -5e-4, 1e-4, 8e-4, -1e-4, np.nan])
    assert build_signal(fc, 2.0).tolist() == [0, -1, 0, 1, -1, 0]
    assert build_signal(fc, 2.0, "reversion").tolist() == [0, -1, 0, 1, -1, 0]
    assert build_signal(fc, 2.0, min_edge_bps=3).tolist() == [0, -1, 0, 1, 0, 0]


def test_backtest_end_to_end_accounting():
    z = [np.nan, np.nan, 3.0, 0, 0, 0, -3.0, 0, 3.0, 0, 0, 0, 0]
    r_hat = [np.nan, np.nan] + [-1e-3 if v > 0 else 1e-3 if v < 0 else 0.0 for v in z[2:]]
    fc = _fc(z, r_hat)
    rng = np.random.default_rng(0)
    exret = pd.Series(rng.normal(0, 0.01, len(z)), index=fc.index)
    res = backtest(fc, exret, k=2.0, H=3, policy="B_reset_flip", cost_bps=5)
    d = res.daily
    assert d["pos"].tolist() == [0, 0, -1, -1, -1, 0, 1, 1, -1, -1, -1, 0, 0]
    assert d["cost"].sum() == pytest.approx(6 * 5e-4)  # 3 entries + 2 exits + flip counted as 2 = 6 units
    assert d["net"].iloc[:3].isna().all() and d["net"].iloc[3:].notna().all()  # warm-up
    assert d["net"].iloc[3] == pytest.approx(-exret.iloc[3] - 5e-4)
    tr = res.trades
    assert tr["direction"].tolist() == [-1, 1, -1]
    assert tr["days_held"].tolist() == [3, 2, 3]
    assert tr["exit_reason"].tolist() == ["expiry", "flip", "expiry"]
    assert tr["gross"].sum() == pytest.approx(d["gross"].sum())
    assert tr["cost"].sum() == pytest.approx(d["cost"].sum())
    assert tr["net"].sum() == pytest.approx(d["net"].sum())


def test_rolled_trade_cost_not_double_counted():
    # A_run_out, same-direction signal on expiry day: position unchanged -> no cost
    zs = [3, 0, 0, 3, 0, 0, 0, 0]
    fc = _fc([np.nan] + zs, [np.nan] + [-1e-3 if v else 0 for v in zs])
    exret = pd.Series(0.001, index=fc.index)
    res = backtest(fc, exret, k=2, H=3, policy="A_run_out", cost_bps=10)
    assert res.daily["pos"].tolist() == [0, -1, -1, -1, -1, -1, -1, 0, 0]
    assert len(res.trades) == 2
    assert res.trades["cost"].sum() == pytest.approx(res.daily["cost"].sum()) == pytest.approx(2e-3)


def test_exit_on_final_close_not_charged():
    fc = _fc([np.nan, 3, 0, 0], [np.nan, -1e-3, 0, 0])
    res = backtest(fc, pd.Series(0.0, index=fc.index), k=2, H=2, policy="A_run_out", cost_bps=10)
    assert res.daily["pos"].tolist() == [0, -1, -1, 0]
    assert res.trades["cost"].sum() == pytest.approx(res.daily["cost"].sum()) == pytest.approx(1e-3)
