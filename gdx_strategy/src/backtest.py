"""Signal -> position engine -> daily P&L and trade list.

Timing (row t = close of day t):
  * signal_t is formed from z_t / r_hat_{t+1}, both known at the close of t;
  * the trade executes at that same close, so pos_t is the position held AFTER
    the close of t;
  * strategy return on row t+1:  strat_{t+1} = pos_t * ExRet_GDX_{t+1} - cost_{t+1},
    where cost_{t+1} = cost_bps * |pos_t - pos_{t-1}| is the cost of the trade done
    at close t (booked on the first bar that position is held);
  * a trade entered (or clock-reset) at close t is closed at close t+H, i.e. it
    earns the H returns t+1 .. t+H, unless the policy ends or extends it.

Order of events at each close: (1) a trade whose clock hits H is closed;
(2) the day's signal is applied. A signal on the expiry day therefore opens a
new trade (in the same direction the position is simply kept, at no cost).

Returns are a self-financing overlay on 1x notional, in excess of cash.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - plain-Python fallback
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

POLICIES = {"A_run_out": 0, "B_reset_flip": 1, "C_reset_flat": 2, "D_stack_2x": 3}
SELECTABLE_POLICIES = ("A_run_out", "B_reset_flip", "C_reset_flat")  # D is reference-only

# exit codes written on the close where a trade ends
EXIT_NONE, EXIT_EXPIRY, EXIT_FLIP, EXIT_SIGNAL_FLAT = 0, 1, 2, 3
EXIT_NAMES = {EXIT_EXPIRY: "expiry", EXIT_FLIP: "flip", EXIT_SIGNAL_FLAT: "opposite_signal_flat"}


@njit(cache=True)
def _engine(sig, H, policy):
    """Path-dependent position loop.

    sig    : int8[N] in {-1, 0, +1}, already gated (|z| >= k, edge filter)
    H      : holding period in days (>= 1)
    policy : 0 A_run_out, 1 B_reset_flip, 2 C_reset_flat, 3 D_stack_2x
    Returns pos[N], trade_id[N] (0 when flat), new_trade[N], exit_code[N].
    """
    N = sig.shape[0]
    pos_out = np.zeros(N, np.int8)
    tid_out = np.zeros(N, np.int32)
    new_out = np.zeros(N, np.int8)
    exit_out = np.zeros(N, np.int8)
    pos = 0
    age = 0
    tid = 0
    for t in range(N):
        # (1) expiry
        if pos != 0:
            age += 1
            if age >= H:
                pos = 0
                exit_out[t] = 1
        # (2) signal
        s = sig[t]
        if s != 0:
            if pos == 0:
                pos = s
                age = 0
                tid += 1
                new_out[t] = 1
            elif (pos > 0) == (s > 0):  # same direction
                if policy != 0:
                    age = 0
                    if policy == 3 and (pos == 1 or pos == -1):
                        pos = 2 * s
            else:  # opposite direction
                if policy == 1 or policy == 3:
                    pos = s
                    age = 0
                    tid += 1
                    new_out[t] = 1
                    exit_out[t] = 2
                elif policy == 2:
                    pos = 0
                    exit_out[t] = 3
                # policy 0: ignore
        pos_out[t] = pos
        tid_out[t] = tid if pos != 0 else 0
    return pos_out, tid_out, new_out, exit_out


def run_engine(signal, H: int, policy: str):
    """Thin wrapper: validates inputs and calls the compiled loop."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {list(POLICIES)}")
    H = int(H)
    if H < 1:
        raise ValueError("H must be >= 1")
    sig = np.asarray(signal)
    if not np.isin(sig, (-1, 0, 1)).all():
        raise ValueError("signal must be in {-1, 0, +1}")
    return _engine(sig.astype(np.int8), H, POLICIES[policy])


def build_signal(fc: pd.DataFrame, k: float, direction_mode: str = "estimated",
                 min_edge_bps: float = 0.0) -> pd.Series:
    """Entry signal at close t: direction if |z_t| >= k (and |r_hat| >= min_edge), else 0."""
    from src.forecast import trade_direction

    d = trade_direction(fc, direction_mode)
    gate = fc["z"].abs() >= k
    if min_edge_bps > 0:
        gate &= fc["r_hat"].abs() * 1e4 >= min_edge_bps
    return d.where(gate.fillna(False), 0).astype(np.int8).rename("signal")


def strategy_returns(pos: np.ndarray, r: np.ndarray, cost_bps: float):
    """(gross, cost, net) aligned to row t+1; row 0 is 0. Missing returns count as 0."""
    pos = pos.astype(float)
    r = np.nan_to_num(np.asarray(r, float), nan=0.0)
    turnover = np.abs(np.diff(np.r_[0.0, pos]))  # units traded at close t
    gross = np.r_[0.0, pos[:-1] * r[1:]]
    cost = np.r_[0.0, turnover[:-1]] * cost_bps * 1e-4
    return gross, cost, gross - cost


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    params: dict = field(default_factory=dict)

    @property
    def net(self) -> pd.Series:
        return self.daily["net"].dropna()

    @property
    def gross(self) -> pd.Series:
        return self.daily["gross"].dropna()


def _trade_table(daily: pd.DataFrame, r_next: np.ndarray, cost_bps: float) -> pd.DataFrame:
    pos = daily["pos"].to_numpy(int)
    tid = daily["trade_id"].to_numpy()
    ex = daily["exit_code"].to_numpy()
    dates = daily.index
    n = int(tid.max()) if len(tid) else 0
    if n == 0:
        return pd.DataFrame(columns=["trade_id", "entry_date", "exit_date", "direction", "max_units",
                                     "days_held", "exit_reason", "gross", "cost", "net"])
    held = tid > 0
    pnl = np.bincount(tid[held], weights=pos[held] * r_next[held], minlength=n + 1)
    days = np.bincount(tid[held], minlength=n + 1)
    maxu = np.zeros(n + 1)
    np.maximum.at(maxu, tid[held], np.abs(pos[held]))

    # cost attribution: same trade continuing -> |dpos| to it; roll into a same-sign new
    # trade (expiry + re-entry) -> |dpos| to the new one; otherwise close old, open new.
    # A change on the final close has no later bar to book its cost on, so (as in the
    # daily series) it is not charged.
    units = np.zeros(n + 1)
    p_prev, t_prev = np.r_[0, pos[:-1]], np.r_[0, tid[:-1]]
    changes = np.flatnonzero((pos != p_prev) | (tid != t_prev))
    for i in changes[changes < len(pos) - 1]:
        if tid[i] == t_prev[i] or (tid[i] and t_prev[i] and np.sign(pos[i]) == np.sign(p_prev[i])):
            units[tid[i] or t_prev[i]] += abs(pos[i] - p_prev[i])
        else:
            units[t_prev[i]] += abs(p_prev[i])
            units[tid[i]] += abs(pos[i])
    units[0] = 0

    first = pd.Series(np.arange(len(tid))[held]).groupby(tid[held]).agg(["first", "last"])
    rows = []
    for i in range(1, n + 1):
        f, l = first.loc[i, "first"], first.loc[i, "last"]
        close_i = l + 1 if l + 1 < len(tid) else None  # close on which it was exited
        reason = EXIT_NAMES[int(ex[close_i])] if close_i is not None else "open"
        rows.append(dict(trade_id=i, entry_date=dates[f],
                         exit_date=dates[close_i] if close_i is not None else pd.NaT,
                         direction=int(np.sign(pos[f])), max_units=int(maxu[i]),
                         days_held=int(days[i]), exit_reason=reason, gross=pnl[i],
                         cost=units[i] * cost_bps * 1e-4))
    tr = pd.DataFrame(rows)
    tr["net"] = tr["gross"] - tr["cost"]
    return tr


def backtest(fc: pd.DataFrame, exret: pd.Series, k: float, H: int, policy: str,
             direction_mode: str = "estimated", min_edge_bps: float = 0.0,
             cost_bps: float = 2.0) -> BacktestResult:
    """Full backtest with daily frame and trade list.

    Rows up to and including the first day a forecast exists are warm-up: their
    gross/net are NaN so they don't enter performance statistics. After that,
    flat days count as zero return.
    """
    exret = exret.reindex(fc.index)
    sig = build_signal(fc, k, direction_mode, min_edge_bps)
    pos, tid, new, ex = run_engine(sig.to_numpy(), H, policy)
    r = exret.to_numpy(float)
    gross, cost, net = strategy_returns(pos, r, cost_bps)
    daily = pd.DataFrame({"signal": sig.to_numpy(), "pos": pos, "trade_id": tid, "new_trade": new,
                          "exit_code": ex, "exret": r, "gross": gross, "cost": cost, "net": net},
                         index=fc.index)
    first = fc["r_hat"].first_valid_index()
    live = np.zeros(len(daily), bool) if first is None else daily.index > first
    daily.loc[~live, ["gross", "cost", "net"]] = np.nan
    r_next = np.r_[np.nan_to_num(r[1:], nan=0.0), 0.0]
    trades = _trade_table(daily, r_next, cost_bps)
    params = dict(k=k, H=H, policy=policy, direction_mode=direction_mode,
                  min_edge_bps=min_edge_bps, cost_bps=cost_bps)
    return BacktestResult(daily, trades, params)
