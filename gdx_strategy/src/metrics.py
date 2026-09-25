"""Performance, frequency, consistency and exposure metrics, vectorised over configs.

Conventions
-----------
* Inputs are positions ``pos[c, t]`` (held after close t) and new-trade flags, as
  stored by the grid. Daily net return on row t:
      net_t = pos_{t-1} * ExRet_GDX_t - cost_bps * |pos_{t-1} - pos_{t-2}|
  (1x notional, self-financing overlay, in excess of cash; flat days are 0).
* ``mask`` selects the scored P&L rows of a period. A close t belongs to the
  period when its next row t+1 does, so entries/turnover are counted on the
  closes whose P&L lands in the period.
* Trades are assigned to the period of their entry close; their P&L runs to
  exit, even past the period end.
* Cumulative P&L and drawdowns are additive (constant 1x notional).
* Calendar months with fewer than MIN_DAYS_PER_MONTH closes in the period
  (partial months at the period edges) are ignored in the month coverage.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

ANN = 252
ROLL_WINDOW = 252
MIN_DAYS_PER_MONTH = 10
TOP_DAY_FRAC = 0.01
TOP_TRADES = 10
EXPOSURE_ASSETS = ("GDX", "SPY", "GLD")


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@njit(cache=True)
def _trades(pos, new, r_next):
    """Per-trade entry index, direction, days held, gross P&L and units traded.

    Cost attribution matches backtest._trade_table: same trade continuing ->
    |dpos| to it; roll into a same-sign new trade -> |dpos| to the new one;
    otherwise |old| closes the old trade and |new| opens the new one. A change
    on the final close is not charged (no later bar to book it on).
    """
    T = pos.shape[0]
    n = 0
    for t in range(T):
        n += new[t]
    entry = np.empty(n, np.int64)
    dirn = np.empty(n, np.int8)
    days = np.zeros(n, np.int64)
    gross = np.zeros(n)
    units = np.zeros(n)
    k = -1
    prev = 0
    for t in range(T):
        p = int(pos[t])
        d = abs(p - prev)
        if t < T - 1:
            if new[t]:
                if k >= 0 and prev != 0 and (prev > 0) == (p > 0):
                    units[k + 1] += d
                else:
                    if k >= 0 and prev != 0:
                        units[k] += abs(prev)
                    units[k + 1] += abs(p)
            elif d > 0 and k >= 0:
                units[k] += d
        if new[t]:
            k += 1
            entry[k] = t
            dirn[k] = 1 if p > 0 else -1
        if p != 0 and k >= 0:
            gross[k] += p * r_next[t]
            days[k] += 1
        prev = p
    return entry, dirn, days, gross, units


def trade_list(pos: np.ndarray, new: np.ndarray, exret: np.ndarray, cost_bps: float) -> pd.DataFrame:
    """Trades of one config: entry index, direction, days, gross, cost, net."""
    r_next = np.r_[np.nan_to_num(exret[1:], nan=0.0), 0.0]
    e, d, h, g, u = _trades(pos.astype(np.int8), new.astype(np.int8), r_next)
    cost = u * cost_bps * 1e-4
    return pd.DataFrame(dict(entry_idx=e, direction=d, days_held=h, gross=g, cost=cost, net=g - cost))


# ---------------------------------------------------------------------------
# Helpers on [configs x days] matrices
# ---------------------------------------------------------------------------

def daily_net(pos: np.ndarray, exret: np.ndarray, cost_bps: float) -> np.ndarray:
    """[n, T] net returns (row 0 = 0)."""
    p = pos.astype(np.float64)
    r = np.nan_to_num(exret, nan=0.0)
    prev = np.zeros_like(p)
    prev[:, 1:] = p[:, :-1]
    turn = np.abs(np.diff(np.concatenate([np.zeros((len(p), 1)), p], axis=1), axis=1))
    turn_prev = np.zeros_like(p)
    turn_prev[:, 1:] = turn[:, :-1]
    return prev * r[None, :] - turn_prev * cost_bps * 1e-4


def _sharpe(x: np.ndarray, axis=-1) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return x.mean(axis) / x.std(axis, ddof=1) * np.sqrt(ANN)


def newey_west_t(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """t-stat of the mean with Bartlett-kernel HAC variance, per-row lag count."""
    n, T = x.shape
    xc = x - x.mean(1, keepdims=True)
    lrv = (xc * xc).mean(1)
    Lmax = int(lags.max()) if len(lags) else 0
    for l in range(1, Lmax + 1):
        g = (xc[:, l:] * xc[:, :-l]).sum(1) / T
        w = np.where(l <= lags, 1.0 - l / (lags + 1.0), 0.0)
        lrv = lrv + 2.0 * w * g
    with np.errstate(invalid="ignore", divide="ignore"):
        return x.mean(1) / np.sqrt(np.maximum(lrv, 0.0) / T)


def _rolling_sharpe(x: np.ndarray, w: int) -> np.ndarray:
    """[n, T-w+1] rolling Sharpe over full windows."""
    n, T = x.shape
    if T < w:
        return np.full((n, 0), np.nan)
    c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(x, 1)], 1)
    c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(x * x, 1)], 1)
    s1 = c1[:, w:] - c1[:, :-w]
    s2 = c2[:, w:] - c2[:, :-w]
    mean = s1 / w
    var = np.maximum((s2 - w * mean * mean) / (w - 1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return mean / np.sqrt(var) * np.sqrt(ANN)


def _segments(labels: np.ndarray):
    """Start indices and labels of runs of equal, sorted labels."""
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    return starts, labels[starts]


def _beta_corr(y: np.ndarray, x: np.ndarray):
    xc = x - x.mean()
    yc = y - y.mean(1, keepdims=True)
    cov = (yc * xc[None, :]).sum(1) / (len(x) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cov / xc.var(ddof=1)
        corr = cov / (yc.std(1, ddof=1) * xc.std(ddof=1))
    return beta, corr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class PeriodMetrics:
    table: pd.DataFrame          # one row per config
    yearly_pnl: pd.DataFrame     # configs x calendar years, summed net return
    yearly_sharpe: pd.DataFrame  # configs x calendar years


def compute_metrics(pos: np.ndarray, new: np.ndarray, dates: pd.DatetimeIndex, returns: pd.DataFrame,
                    mask: np.ndarray, cost_bps: float, H: np.ndarray, ids=None,
                    with_trades: bool = True) -> PeriodMetrics:
    """Metrics for each config (row of ``pos``) over the rows in ``mask``.

    ``returns`` must contain ExRet_GDX (and ExRet_SPY / ExRet_GLD for exposures),
    aligned to ``dates``.
    """
    pos = np.atleast_2d(pos)
    new = np.atleast_2d(new)
    n, T = pos.shape
    H = np.broadcast_to(np.asarray(H, float), (n,))
    ids = pd.Index(range(n) if ids is None else ids, name="config_id")
    mask = np.asarray(mask, bool)
    rows = np.flatnonzero(mask)
    if len(rows) < 2:
        raise ValueError("period has fewer than 2 scored rows")
    close_mask = np.zeros(T, bool)
    close_mask[rows[rows > 0] - 1] = True  # close t is in the period if row t+1 is
    exret = returns["ExRet_GDX"].to_numpy(float)
    d = dates[mask]
    Tm = len(rows)
    yrs = Tm / ANN

    net_full = daily_net(pos, exret, cost_bps)
    gross_full = daily_net(pos, exret, 0.0)
    net, gross = net_full[:, mask], gross_full[:, mask]
    prev_pos = np.zeros_like(pos, dtype=np.float64)
    prev_pos[:, 1:] = pos[:, :-1]
    prev_pos = prev_pos[:, mask]
    out: dict[str, np.ndarray] = {}

    # -- performance ---------------------------------------------------------
    mu = net.mean(1)
    sd = net.std(1, ddof=1)
    out["ann_return"] = mu * ANN
    out["ann_vol"] = sd * np.sqrt(ANN)
    out["sharpe"] = _sharpe(net, 1)
    out["sharpe_gross"] = _sharpe(gross, 1)
    down = np.sqrt((np.minimum(net, 0.0) ** 2).mean(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["sortino"] = mu / down * np.sqrt(ANN)
    cum = np.cumsum(net, 1)
    dd = cum - np.maximum.accumulate(np.maximum(cum, 0.0), 1)
    out["max_drawdown"] = dd.min(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["calmar"] = np.where(out["max_drawdown"] < 0, out["ann_return"] / -out["max_drawdown"], np.nan)
    out["nw_tstat"] = newey_west_t(net, H)
    out["day_hit_rate"] = (net > 0).sum(1) / np.maximum((prev_pos != 0).sum(1), 1)

    # -- frequency -----------------------------------------------------------
    new_in = new[:, close_mask].astype(np.int64)
    out["n_entries"] = new_in.sum(1)
    months = dates[close_mask].to_period("M").asi8
    m_starts, _ = _segments(months)
    m_len = np.diff(np.r_[m_starts, len(months)])
    per_month = np.add.reduceat(new_in, m_starts, axis=1)[:, m_len >= MIN_DAYS_PER_MONTH]
    n_months = per_month.shape[1]
    out["entries_per_month"] = out["n_entries"] / max(n_months, 1)
    out["pct_months_with_entry"] = (per_month >= 1).mean(1) if n_months else np.nan
    out["min_entries_in_month"] = per_month.min(1) if n_months else np.nan
    out["pct_days_in_market"] = (prev_pos != 0).mean(1)
    dpos = np.abs(np.diff(np.concatenate([np.zeros((n, 1)), pos.astype(float)], 1), axis=1))
    out["turnover_per_year"] = dpos[:, close_mask].sum(1) / yrs

    # -- consistency -----------------------------------------------------------
    y_starts, y_labels = _segments(np.asarray(d.year))
    yearly = np.add.reduceat(net, y_starts, axis=1)
    y_len = np.diff(np.r_[y_starts, Tm])
    yearly_sharpe = np.column_stack([_sharpe(net[:, s:s + L], 1) for s, L in zip(y_starts, y_len)])
    out["pct_years_positive"] = (yearly > 0).mean(1)
    out["worst_year"] = yearly.min(1)
    with warnings.catch_warnings():  # configs flat for a whole year have NaN Sharpe there
        warnings.simplefilter("ignore", RuntimeWarning)
        out["worst_year_sharpe"] = np.nanmin(yearly_sharpe, 1)
        out["median_year_sharpe"] = np.nanmedian(yearly_sharpe, 1)
    rs = _rolling_sharpe(net, ROLL_WINDOW)
    out["rolling12m_sharpe_min"] = rs.min(1) if rs.shape[1] else np.nan
    out["rolling12m_pct_positive"] = (rs > 0).mean(1) if rs.shape[1] else np.nan
    half = Tm // 2
    out["sharpe_first_half"] = _sharpe(net[:, :half], 1)
    out["sharpe_second_half"] = _sharpe(net[:, half:], 1)
    out["long_leg_pnl"] = np.where(prev_pos > 0, gross, 0.0).sum(1) / yrs
    out["short_leg_pnl"] = np.where(prev_pos < 0, gross, 0.0).sum(1) / yrs
    k_top = max(1, int(np.ceil(TOP_DAY_FRAC * Tm)))
    top_days = -np.partition(-net, k_top - 1, axis=1)[:, :k_top].sum(1)
    out["ann_return_ex_top1pct_days"] = (net.sum(1) - top_days) / yrs

    # -- exposure --------------------------------------------------------------
    for a in EXPOSURE_ASSETS:
        col = f"ExRet_{a}"
        if col in returns:
            b, c = _beta_corr(net, np.nan_to_num(returns[col].to_numpy(float)[mask]))
            out[f"beta_{a}"], out[f"corr_{a}"] = b, c

    # -- trade-level -------------------------------------------------------------
    if with_trades:
        r_next = np.r_[np.nan_to_num(exret[1:], nan=0.0), 0.0]
        keys = ["hit_rate", "avg_win", "avg_loss", "profit_factor", "mean_trade", "avg_holding_days",
                "top10_trades_pnl_share"]
        tv = {k: np.full(n, np.nan) for k in keys}
        for i in range(n):
            e, _, h, g, u = _trades(pos[i].astype(np.int8), new[i].astype(np.int8), r_next)
            keep = close_mask[e]
            if not keep.any():
                continue
            tn = g[keep] - u[keep] * cost_bps * 1e-4
            wins, losses = tn[tn > 0], tn[tn <= 0]
            tv["hit_rate"][i] = len(wins) / len(tn)
            tv["avg_win"][i] = wins.mean() if len(wins) else np.nan
            tv["avg_loss"][i] = losses.mean() if len(losses) else np.nan
            tv["profit_factor"][i] = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
            tv["mean_trade"][i] = tn.mean()
            tv["avg_holding_days"][i] = h[keep].mean()
            total = tn.sum()
            tv["top10_trades_pnl_share"][i] = (np.sort(tn)[-TOP_TRADES:].sum() / total) if total > 0 else np.nan
        out.update(tv)

    table = pd.DataFrame(out, index=ids)
    table["n_days"] = Tm
    table["n_months"] = n_months
    cols = pd.Index(y_labels, name="year")
    return PeriodMetrics(table, pd.DataFrame(yearly, index=ids, columns=cols),
                         pd.DataFrame(yearly_sharpe, index=ids, columns=cols))


def grid_metrics(grid, returns: pd.DataFrame, mask: np.ndarray, cost_bps: float, rows=None,
                 chunk: int = 1500, with_trades: bool = True) -> PeriodMetrics:
    """compute_metrics for (a subset of) a GridResult, in memory-bounded chunks."""
    cfg = grid.configs if rows is None else grid.configs.loc[rows]
    idx = grid.configs.index.get_indexer(cfg.index)
    returns = returns.reindex(grid.dates)
    parts = [compute_metrics(grid.pos[idx[s:s + chunk]], grid.new_trade[idx[s:s + chunk]], grid.dates,
                             returns, mask, cost_bps, cfg["H"].to_numpy()[s:s + chunk],
                             cfg.index[s:s + chunk], with_trades)
             for s in range(0, len(idx), chunk)]
    return PeriodMetrics(*(pd.concat([getattr(p, f) for p in parts])
                           for f in ("table", "yearly_pnl", "yearly_sharpe")))
