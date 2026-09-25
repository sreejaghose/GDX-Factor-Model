"""Selection: hard filters, neighbourhood robustness score, statistical checks,
and the selection rule shared by the development shortlist and walk-forward.
"""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.grid import GridResult
from src.metrics import ANN, grid_metrics, trade_list

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

log = logging.getLogger(__name__)

CATEGORICAL = ["factors", "policy", "direction_mode"]
NUMERIC = ["lookback", "m", "k", "H"]
FILTERS = ["f_months", "f_entries", "f_halves", "f_years", "f_ex_top_days", "f_policy"]


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------

def hard_filters(table: pd.DataFrame, configs: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """Boolean filter columns + 'passes' + 'n_failed', aligned to table.index.

    min_entries is scaled by (period days / min_entries_ref_days) so the same
    rule applies to shorter windows (validation, walk-forward training sets).
    """
    c = configs.loc[table.index]
    min_entries = sel["min_entries"] * table["n_days"] / sel["min_entries_ref_days"]
    f = pd.DataFrame(index=table.index)
    f["f_months"] = table["pct_months_with_entry"] >= sel["min_pct_months_with_entry"] - 1e-12
    f["f_entries"] = table["n_entries"] >= np.floor(min_entries)
    f["f_halves"] = ((table["sharpe_first_half"] > 0) & (table["sharpe_second_half"] > 0)) \
        if sel.get("require_positive_halves", True) else True
    f["f_years"] = table["pct_years_positive"] >= sel["min_pct_years_positive"] - 1e-12
    f["f_ex_top_days"] = (table["ann_return_ex_top1pct_days"] > 0) \
        if sel.get("require_positive_ex_top_days", True) else True
    f["f_policy"] = c["policy"] != "D_stack_2x"
    f["n_failed"] = (~f[FILTERS]).sum(axis=1)
    f["passes"] = f["n_failed"] == 0
    return f


def filter_funnel(filters: pd.DataFrame) -> pd.DataFrame:
    """How many configs survive each filter, applied cumulatively in order."""
    alive = pd.Series(True, index=filters.index)
    rows = [("all", len(alive))]
    for name in FILTERS:
        alive &= filters[name]
        rows.append((name, int(alive.sum())))
    return pd.DataFrame(rows, columns=["after_filter", "n_configs"])


# ---------------------------------------------------------------------------
# Neighbourhood robustness
# ---------------------------------------------------------------------------

def neighbourhood_scores(value: pd.Series, configs: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """Neighbourhood stats of ``value`` (e.g. dev net Sharpe) for every config.

    Neighbourhood = same factors / policy / direction_mode, and every numeric
    param (L, m, k, H) at its own grid value or one step either side: up to
    3^4 = 81 configs including itself (fewer at grid edges).
    """
    c = configs.loc[value.index]
    axes = [list(grid[p]) for p in NUMERIC]
    shape = tuple(len(a) for a in axes)
    pos_idx = [c[p].map({v: i for i, v in enumerate(a)}).to_numpy() for p, a in zip(NUMERIC, axes)]
    if any(np.isnan(np.asarray(ix, float)).any() for ix in pos_idx):
        raise ValueError("config value not on the grid")
    out = pd.DataFrame(index=value.index, columns=["nbhd_median", "nbhd_mean", "nbhd_min",
                                                    "nbhd_frac_positive", "nbhd_n"], dtype=float)
    keys = c[CATEGORICAL].astype(str).agg("|".join, axis=1)
    for key, rows in keys.groupby(keys).groups.items():
        loc = value.index.get_indexer(rows)
        arr = np.full(tuple(s + 2 for s in shape), np.nan)
        coords = tuple(ix[loc] + 1 for ix in pos_idx)
        arr[coords] = value.to_numpy()[loc]
        stack = np.stack([arr[tuple(slice(1 + o, 1 + o + s) for o, s in zip(off, shape))]
                          for off in itertools.product((-1, 0, 1), repeat=4)])
        inner = tuple(ix[loc] for ix in pos_idx)
        nb = stack[(slice(None),) + inner]  # [81, n_rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out.iloc[loc, 0] = np.nanmedian(nb, 0)
            out.iloc[loc, 1] = np.nanmean(nb, 0)
            out.iloc[loc, 2] = np.nanmin(nb, 0)
            out.iloc[loc, 3] = np.nansum(nb > 0, 0) / np.sum(~np.isnan(nb), 0)
            out.iloc[loc, 4] = np.sum(~np.isnan(nb), 0)
    return out


# ---------------------------------------------------------------------------
# Ranking / selection rule
# ---------------------------------------------------------------------------

def simplicity_key(configs: pd.DataFrame) -> pd.DataFrame:
    """Tie-break order: fewer factors, shorter H, policy B first."""
    return pd.DataFrame({"n_factors": configs["factors"].map(len),
                         "H": configs["H"],
                         "policy_rank": configs["policy"].map({"B_reset_flip": 0, "A_run_out": 1,
                                                               "C_reset_flat": 2, "D_stack_2x": 3})},
                        index=configs.index)


def rank_configs(grid: GridResult, returns: pd.DataFrame, mask: np.ndarray, grid_cfg: dict,
                 with_trades: bool = False) -> pd.DataFrame:
    """Metrics + filters + neighbourhood score for every selectable config, ranked.

    Ranking: passing configs first, then fewest failed filters; within that,
    neighbourhood median Sharpe (desc), then simplicity.
    """
    sel = grid_cfg["selection"]
    ids = grid.configs.index[grid.configs["policy"] != "D_stack_2x"]
    pm = grid_metrics(grid, returns, mask, grid_cfg["fixed"]["cost_bps"], rows=ids, with_trades=with_trades)
    t = pm.table
    f = hard_filters(t, grid.configs, sel)
    nb = neighbourhood_scores(t["sharpe"], grid.configs, grid_cfg["grid"])
    df = pd.concat([grid.configs.loc[ids], t, f, nb,
                    simplicity_key(grid.configs.loc[ids]).drop(columns="H")], axis=1)
    df = df.sort_values(["n_failed", "nbhd_median", "n_factors", "H", "policy_rank"],
                        ascending=[True, False, True, True, True])
    df["rank"] = np.arange(1, len(df) + 1)
    df.attrs = {}
    return df


def make_selection_rule(returns: pd.DataFrame, grid_cfg: dict):
    """The rule used for the dev shortlist, applied as rule(grid, mask) -> config_id."""
    def rule(grid: GridResult, mask: np.ndarray) -> str:
        ranked = rank_configs(grid, returns, mask, grid_cfg)
        top = ranked.iloc[0]
        if not top["passes"]:
            log.warning("No config passes all hard filters on window ending %s; taking best with "
                        "%d failed filter(s)", grid.dates[mask][-1].date(), int(top["n_failed"]))
        return top.name
    return rule


# ---------------------------------------------------------------------------
# Statistical checks
# ---------------------------------------------------------------------------

def deflated_sharpe(net: np.ndarray, sr_trials_daily: np.ndarray, n_trials: int) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014), daily units.

    SR0 = sqrt(Var[SR_trials]) * ((1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))
    DSR = Z( (SR - SR0) sqrt(T-1) / sqrt(1 - skew SR + (kurt-1)/4 SR^2) )
    """
    T = len(net)
    sr = net.mean() / net.std(ddof=1)
    g = 0.5772156649015329
    v = np.nanvar(sr_trials_daily, ddof=1)
    sr0 = np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials)
                        + g * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    sk = stats.skew(net)
    ku = stats.kurtosis(net, fisher=False)
    denom = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    psr0 = stats.norm.cdf(sr * np.sqrt(T - 1) / denom)  # probabilistic SR vs 0
    return dict(dsr=float(dsr), psr_vs_zero=float(psr0), sr_ann=float(sr * np.sqrt(ANN)),
                sr0_ann=float(sr0 * np.sqrt(ANN)), n_trials=int(n_trials), skew=float(sk), kurtosis=float(ku))


@njit(cache=True)
def _placebo_positions(T, lengths, dirs, n_sims, seed):
    """Random non-overlapping trades with the same count, lengths and directions."""
    np.random.seed(seed)
    n = lengths.shape[0]
    out = np.zeros((n_sims, T), np.int8)
    free = T - lengths.sum()
    for s in range(n_sims):
        order = np.random.permutation(n)
        cuts = np.sort(np.random.randint(0, free + 1, n))
        t = 0
        last = 0
        for j in range(n):
            t += cuts[j] - last
            last = cuts[j]
            i = order[j]
            for u in range(lengths[i]):
                out[s, t + u] = dirs[i]
            t += lengths[i]
    return out


def _local_net(P: np.ndarray, r: np.ndarray, cost_bps: float, p0: float = 0.0) -> np.ndarray:
    """net_i = P_i * r_i - cost * |P_i - P_{i-1}|  for P [..., T] held into row i.

    ``p0`` is the position held before the window's first close.
    """
    prev = np.full_like(P, p0, dtype=float)
    prev[..., 1:] = P[..., :-1]
    return P * r - np.abs(P - prev) * cost_bps * 1e-4


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return x.mean(-1) / x.std(-1, ddof=1) * np.sqrt(ANN)


@njit(cache=True)
def _stationary_bootstrap_sharpe(x, reps, block, seed):
    np.random.seed(seed)
    T = x.shape[0]
    p = 1.0 / block
    out = np.empty(reps)
    for b in range(reps):
        s1 = 0.0
        s2 = 0.0
        i = np.random.randint(0, T)
        for t in range(T):
            if t > 0:
                if np.random.random() < p:
                    i = np.random.randint(0, T)
                else:
                    i = (i + 1) % T
            v = x[i]
            s1 += v
            s2 += v * v
        m = s1 / T
        var = (s2 - T * m * m) / (T - 1)
        out[b] = m / np.sqrt(var) * np.sqrt(252.0) if var > 0 else np.nan
    return out


def statistical_checks(grid: GridResult, cid: str, mask: np.ndarray, cost_bps: float,
                       sr_trials_daily: np.ndarray, n_trials: int, st: dict) -> dict:
    """DSR, random-entry placebo, sign flip and stationary-bootstrap CI for one config."""
    i = grid.configs.index.get_loc(cid)
    rows = np.flatnonzero(mask)
    closes = rows - 1                              # P_i = pos held into scored row i
    P = grid.pos[i, closes].astype(float)
    r = np.nan_to_num(grid.exret[rows], nan=0.0)
    p0 = float(grid.pos[i, closes[0] - 1])
    net = _local_net(P, r, cost_bps, p0)   # identical to the grid's daily net on these rows
    out = {"sharpe": float(_sharpe_rows(net))}
    out.update({f"dsr_{k}" if k != "dsr" else "dsr": v
                for k, v in deflated_sharpe(net, sr_trials_daily, n_trials).items()})

    # sign flip: same timing, opposite direction
    out["sign_flip_sharpe"] = float(_sharpe_rows(_local_net(-P, r, cost_bps, -p0)))

    # placebo: same number of trades, realised holding lengths and direction mix, placed
    # at random non-overlapping dates in the window (starting flat)
    tl = trade_list(grid.pos[i], grid.new_trade[i], grid.exret, cost_bps)
    in_period = np.isin(tl["entry_idx"].to_numpy(), closes)
    lengths = np.maximum(tl.loc[in_period, "days_held"].to_numpy(np.int64), 1)
    dirs = tl.loc[in_period, "direction"].to_numpy(np.int8)
    excess = lengths.sum() - len(P)
    if excess > 0:  # a trade spilling past the period end: trim the longest ones
        for j in np.argsort(-lengths):
            cut = min(excess, lengths[j] - 1)
            lengths[j] -= cut
            excess -= cut
            if excess <= 0:
                break
    sims = _placebo_positions(len(P), lengths, dirs, int(st["placebo_sims"]), int(st["seed"]))
    sim_sr = _sharpe_rows(_local_net(sims.astype(float), r[None, :], cost_bps))
    out["placebo_n_trades"] = int(len(lengths))
    out["placebo_pct_long"] = float((dirs > 0).mean()) if len(dirs) else np.nan
    out["placebo_mean_sharpe"] = float(np.nanmean(sim_sr))
    out["placebo_p95_sharpe"] = float(np.nanpercentile(sim_sr, 95))
    out["placebo_percentile"] = float((sim_sr < out["sharpe"]).mean() * 100)

    # stationary block bootstrap CI
    bs = _stationary_bootstrap_sharpe(net, int(st["bootstrap_reps"]), float(st["bootstrap_block"]),
                                      int(st["seed"]))
    out["boot_sharpe_lo95"], out["boot_sharpe_hi95"] = (float(v) for v in np.nanpercentile(bs, [2.5, 97.5]))
    out["boot_p_sharpe_le_0"] = float((bs <= 0).mean())
    return out
