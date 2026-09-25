"""Selection machinery: filters, neighbourhood score, DSR, placebo, bootstrap."""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.grid import GridResult
from src.robustness import (_placebo_positions, _stationary_bootstrap_sharpe, deflated_sharpe,
                            hard_filters, neighbourhood_scores, statistical_checks)

SEL = dict(min_pct_months_with_entry=1.0, min_entries=150, min_entries_ref_days=2000,
           min_pct_years_positive=0.7, require_positive_halves=True, require_positive_ex_top_days=True)


def test_hard_filters():
    t = pd.DataFrame({
        "pct_months_with_entry": [1.0, 0.99, 1.0, 1.0, 1.0, 1.0],
        "n_entries": [200, 200, 149, 200, 200, 80],
        "n_days": [2000, 2000, 2000, 2000, 2000, 1000],   # last: half window -> needs 75
        "sharpe_first_half": [0.5, 0.5, 0.5, -0.1, 0.5, 0.5],
        "sharpe_second_half": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "pct_years_positive": [0.75, 0.75, 0.75, 0.75, 0.6, 0.8],
        "ann_return_ex_top1pct_days": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    }, index=list("abcdef"))
    cfg = pd.DataFrame({"policy": ["B_reset_flip"] * 6}, index=t.index)
    f = hard_filters(t, cfg, SEL)
    assert f["passes"].tolist() == [True, False, False, False, False, True]
    assert f.loc["b", "n_failed"] == 1 and not f.loc["b", "f_months"]
    assert not f.loc["c", "f_entries"] and not f.loc["d", "f_halves"] and not f.loc["e", "f_years"]


def _toy_grid():
    grid = dict(lookback=[20, 40, 60], m=[1, 2], k=[0.5, 1.0, 1.5], H=[1, 3])
    rows = [dict(factors=("GLD",), policy=p, direction_mode="estimated", lookback=L, m=m, k=k, H=H)
            for p in ("A_run_out", "B_reset_flip")
            for L, m, k, H in itertools.product(*grid.values())]
    cfg = pd.DataFrame(rows)
    cfg.index = [f"c{i}" for i in range(len(cfg))]
    return grid, cfg


def test_neighbourhood_plateau_vs_peak():
    grid, cfg = _toy_grid()
    val = pd.Series(0.0, index=cfg.index)
    A = cfg["policy"] == "A_run_out"
    # A: isolated peak at the centre; B: plateau of 0.5 everywhere
    peak = cfg.index[A & (cfg.lookback == 40) & (cfg.m == 1) & (cfg.k == 1.0) & (cfg.H == 1)][0]
    val[peak] = 2.0
    val[~A] = 0.5
    nb = neighbourhood_scores(val, cfg, grid)
    assert nb.loc[peak, "nbhd_median"] == 0.0            # isolated peak scores low
    assert nb.loc[peak, "nbhd_n"] == 3 * 2 * 3 * 2       # m, H have only 2 values -> both in range
    assert (nb.loc[~A, "nbhd_median"] == 0.5).all()      # plateau scores its level
    corner = cfg.index[~A & (cfg.lookback == 20) & (cfg.m == 1) & (cfg.k == 0.5) & (cfg.H == 1)][0]
    assert nb.loc[corner, "nbhd_n"] == 2 * 2 * 2 * 2
    # policies never mix
    assert nb.loc[cfg.index[A], "nbhd_mean"].max() < 0.5


def test_deflated_sharpe_formula_and_monotonicity():
    rng = np.random.default_rng(0)
    net = rng.normal(0.001, 0.01, 2000)
    trials = rng.normal(0, 0.02, 500)
    d = deflated_sharpe(net, trials, 500)
    sr = net.mean() / net.std(ddof=1)
    g = 0.5772156649015329
    sr0 = trials.std(ddof=1) * ((1 - g) * stats.norm.ppf(1 - 1 / 500) + g * stats.norm.ppf(1 - 1 / (500 * np.e)))
    den = np.sqrt(1 - stats.skew(net) * sr + (stats.kurtosis(net, fisher=False) - 1) / 4 * sr ** 2)
    assert d["dsr"] == pytest.approx(stats.norm.cdf((sr - sr0) * np.sqrt(1999) / den))
    assert deflated_sharpe(net, trials, 10_000)["dsr"] < d["dsr"] < deflated_sharpe(net, trials, 10)["dsr"]


def test_placebo_positions_preserve_trades():
    lengths = np.array([3, 1, 5, 2], np.int64)
    dirs = np.array([1, -1, -1, 1], np.int8)
    P = _placebo_positions(40, lengths, dirs, 200, 7)
    assert P.shape == (200, 40)
    assert ((P != 0).sum(1) == lengths.sum()).all()
    assert ((P > 0).sum(1) == 5).all() and ((P < 0).sum(1) == 6).all()
    assert not np.array_equal(P[0], P[1])
    np.testing.assert_array_equal(P, _placebo_positions(40, lengths, dirs, 200, 7))  # seeded


def test_bootstrap_centred_on_sample_sharpe():
    x = np.random.default_rng(3).normal(0.0008, 0.01, 3000)
    bs = _stationary_bootstrap_sharpe(x, 2000, 10.0, 1)
    sr = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert abs(np.nanmean(bs) - sr) < 0.15
    lo, hi = np.percentile(bs, [2.5, 97.5])
    assert lo < sr < hi and 0.5 < hi - lo < 2.0


def test_statistical_checks_consistent_with_grid():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2010-01-01", periods=800, name="Date")
    r = rng.normal(0, 0.02, 800)
    pos = np.zeros((1, 800), np.int8)
    new = np.zeros_like(pos)
    t = 10
    while t < 790:  # alternating 3-day trades, one every 5 days
        pos[0, t:t + 3] = 1 if (t // 5) % 2 else -1
        new[0, t] = 1
        t += 5
    cfg = pd.DataFrame(dict(factors=[("GLD",)], lookback=20, m=1, k=1.0, H=3, policy="A_run_out",
                            direction_mode="estimated"), index=pd.Index(["x"], name="config_id"))
    g = GridResult(cfg, dates, r, pos, new, dates[5], 1)
    mask = g.eval_mask & np.asarray(dates >= dates[402])  # starts mid-trade
    st = dict(seed=1, placebo_sims=300, bootstrap_reps=300, bootstrap_block=10)
    out = statistical_checks(g, "x", mask, 5.0, rng.normal(0, 0.03, 100), 100, st)
    net = g.net_returns(5.0, mask=mask).iloc[:, 0]
    assert out["sharpe"] == pytest.approx(net.mean() / net.std() * np.sqrt(252))
    flip = GridResult(cfg, dates, r, -pos, new, dates[5], 1).net_returns(5.0, mask=mask).iloc[:, 0]
    assert out["sign_flip_sharpe"] == pytest.approx(flip.mean() / flip.std() * np.sqrt(252))
    assert 0 <= out["placebo_percentile"] <= 100 and out["placebo_n_trades"] > 50
