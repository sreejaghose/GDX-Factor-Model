"""Metrics vs independent pandas / statsmodels calculations."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.backtest import backtest
from src.metrics import compute_metrics, newey_west_t, trade_list


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2014-01-01", "2018-12-31", name="Date")
    n = len(idx)
    z = pd.Series(rng.normal(size=n), idx)
    z.iloc[:5] = np.nan
    r_hat = -1e-3 * z
    fc = pd.DataFrame({"z": z, "r_hat": r_hat})
    rets = pd.DataFrame({"ExRet_GDX": rng.normal(0, 0.02, n) - 3e-3 * np.r_[0, z.fillna(0).to_numpy()[:-1]],
                         "ExRet_SPY": rng.normal(0, 0.01, n), "ExRet_GLD": rng.normal(0, 0.01, n)}, idx)
    res = backtest(fc, rets["ExRet_GDX"], k=1.0, H=3, policy="B_reset_flip", cost_bps=5)
    return rets, res


def test_trade_list_matches_backtest(case):
    rets, res = case
    d = res.daily
    tl = trade_list(d["pos"].to_numpy(), d["new_trade"].to_numpy(), rets["ExRet_GDX"].to_numpy(), 5)
    tr = res.trades
    assert len(tl) == len(tr)
    np.testing.assert_allclose(tl["gross"], tr["gross"], atol=1e-12)
    np.testing.assert_allclose(tl["cost"], tr["cost"], atol=1e-12)
    np.testing.assert_array_equal(tl["days_held"], tr["days_held"])
    np.testing.assert_array_equal(tl["direction"], tr["direction"])


def test_performance_metrics(case):
    rets, res = case
    d = res.daily
    mask = d["net"].notna().to_numpy()
    pm = compute_metrics(d["pos"].to_numpy()[None], d["new_trade"].to_numpy()[None], d.index,
                         rets, mask, 5.0, np.array([3]))
    m, yearly = pm.table.iloc[0], pm.yearly_pnl
    net, gross = res.net, res.gross
    assert m["ann_return"] == pytest.approx(net.mean() * 252)
    assert m["ann_vol"] == pytest.approx(net.std() * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(net.mean() / net.std() * np.sqrt(252))
    assert m["sharpe_gross"] == pytest.approx(gross.mean() / gross.std() * np.sqrt(252))
    cum = net.cumsum()
    assert m["max_drawdown"] == pytest.approx((cum - cum.cummax().clip(lower=0)).min())
    assert m["sortino"] == pytest.approx(net.mean() / np.sqrt((net.clip(upper=0) ** 2).mean()) * np.sqrt(252))
    # Newey-West vs statsmodels HAC (Bartlett, lags = H, no small-sample correction)
    hac = sm.OLS(net.to_numpy(), np.ones(len(net))).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False})
    assert m["nw_tstat"] == pytest.approx(hac.tvalues[0], rel=1e-9)
    # yearly
    pd.testing.assert_series_equal(yearly.iloc[0], net.groupby(net.index.year).sum(), check_names=False)
    assert m["pct_years_positive"] == pytest.approx((net.groupby(net.index.year).sum() > 0).mean())
    ys = pm.yearly_sharpe.iloc[0]
    g = net.groupby(net.index.year)
    np.testing.assert_allclose(ys.to_numpy(), (g.mean() / g.std() * np.sqrt(252)).to_numpy())
    # rolling 12m
    rs = (net.rolling(252).mean() / net.rolling(252).std() * np.sqrt(252)).dropna()
    assert m["rolling12m_sharpe_min"] == pytest.approx(rs.min(), rel=1e-6)
    assert m["rolling12m_pct_positive"] == pytest.approx((rs > 0).mean())
    # halves
    h = len(net) // 2
    assert m["sharpe_first_half"] == pytest.approx(net.iloc[:h].mean() / net.iloc[:h].std() * np.sqrt(252))
    # legs (gross) add up to total gross
    yrs = len(net) / 252
    assert (m["long_leg_pnl"] + m["short_leg_pnl"]) * yrs == pytest.approx(gross.sum())
    # ex top 1% days
    k = int(np.ceil(0.01 * len(net)))
    assert m["ann_return_ex_top1pct_days"] == pytest.approx((net.sum() - net.nlargest(k).sum()) / yrs)
    # exposures
    x = rets["ExRet_GDX"].loc[net.index]
    assert m["beta_GDX"] == pytest.approx(np.cov(net, x)[0, 1] / x.var())
    assert m["corr_SPY"] == pytest.approx(net.corr(rets["ExRet_SPY"].loc[net.index]))


def test_trade_and_frequency_metrics(case):
    rets, res = case
    d = res.daily
    mask = d["net"].notna().to_numpy()
    m = compute_metrics(d["pos"].to_numpy()[None], d["new_trade"].to_numpy()[None], d.index,
                        rets, mask, 5.0, np.array([3])).table.iloc[0]
    tr = res.trades  # every entry close here is in the period (first entry after warm-up)
    assert m["n_entries"] == len(tr) == d["new_trade"].sum()
    assert m["hit_rate"] == pytest.approx((tr["net"] > 0).mean())
    assert m["mean_trade"] == pytest.approx(tr["net"].mean())
    assert m["avg_win"] == pytest.approx(tr.loc[tr.net > 0, "net"].mean())
    assert m["profit_factor"] == pytest.approx(tr.loc[tr.net > 0, "net"].sum() / -tr.loc[tr.net <= 0, "net"].sum())
    assert m["avg_holding_days"] == pytest.approx(tr["days_held"].mean())
    top = np.sort(tr["net"])[-10:].sum() / tr["net"].sum()
    assert m["top10_trades_pnl_share"] == pytest.approx(top) if tr["net"].sum() > 0 else np.isnan(m["top10_trades_pnl_share"])
    assert m["pct_days_in_market"] == pytest.approx((d["pos"].shift(1).fillna(0) != 0)[mask].mean())
    closes = d.index[np.r_[mask[1:], False]]
    per_m = d["new_trade"].loc[closes].groupby(closes.to_period("M")).agg(["sum", "size"])
    per_m = per_m[per_m["size"] >= 10]
    assert m["pct_months_with_entry"] == pytest.approx((per_m["sum"] >= 1).mean())
    assert m["n_months"] == len(per_m)
    turn = d["pos"].diff().abs().fillna(d["pos"].abs()).loc[closes].sum()
    assert m["turnover_per_year"] == pytest.approx(turn / (mask.sum() / 252))


def test_newey_west_zero_lags_is_plain_t():
    x = np.random.default_rng(0).normal(0.1, 1, (2, 500))
    t = newey_west_t(x, np.array([0, 0]))
    plain = x.mean(1) / (x.std(1, ddof=0) / np.sqrt(500))
    np.testing.assert_allclose(t, plain)
