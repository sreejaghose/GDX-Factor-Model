"""Run the full parameter grid on one dataset.

Stage 1 and Stage 2 are computed once per (F, L, m) and shared by every
(k, direction_mode, H, policy). The engine is ~0.1 ms per config, so the whole
grid runs serially in seconds.

Positions (int8) and new-trade flags are stored for every config, so net returns
can be rebuilt for any cost level and any date slice without rerunning.

Common evaluation window: signals are zeroed before ``signal_start`` (the first
date on which the slowest config has a valid forecast), so every config starts
flat on the same close and P&L is scored from the next row onward.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import POLICIES, build_signal, run_engine
from src.factor_model import ResidualCache, check_factor_set
from src.forecast import ForecastCache

log = logging.getLogger(__name__)

GRID_KEYS = ["factors", "lookback", "m", "k", "H", "policy", "direction_mode"]
POLICY_SHORT = {"A_run_out": "A", "B_reset_flip": "B", "C_reset_flat": "C", "D_stack_2x": "D"}
DIR_SHORT = {"estimated": "est", "reversion": "rev"}


def load_grid(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    g = cfg["grid"]
    missing = [k for k in GRID_KEYS if k not in g]
    if missing:
        raise ValueError(f"grid.yaml missing keys: {missing}")
    g["factors"] = [check_factor_set(f) for f in g["factors"]]
    for p in g["policy"] + cfg.get("reference_only", {}).get("policy", []):
        if p not in POLICIES:
            raise ValueError(f"Unknown policy {p!r}")
    return cfg


def config_id(c: dict) -> str:
    return (f"{'+'.join(c['factors'])}|L{c['lookback']}|m{c['m']}|k{c['k']:g}|H{c['H']}"
            f"|{POLICY_SHORT[c['policy']]}|{DIR_SHORT[c['direction_mode']]}")


def enumerate_configs(grid: dict, policies: list[str] | None = None) -> pd.DataFrame:
    """One row per configuration, indexed by config_id, in a deterministic order."""
    g = dict(grid)
    if policies is not None:
        g["policy"] = policies
    rows = [dict(zip(GRID_KEYS, v)) for v in itertools.product(*(g[k] for k in GRID_KEYS))]
    df = pd.DataFrame(rows)
    df.index = pd.Index([config_id(r) for r in rows], name="config_id")
    if df.index.has_duplicates:
        raise ValueError("Duplicate config ids in grid")
    return df


def n_configs(grid: dict) -> int:
    return int(np.prod([len(grid[k]) for k in GRID_KEYS]))


def common_signal_start(fcache: ForecastCache, grid: dict) -> pd.Timestamp:
    """Latest 'first valid forecast' date over every (F, L, m) in the grid."""
    firsts = []
    for F, L, m in itertools.product(grid["factors"], grid["lookback"], grid["m"]):
        first = fcache.forecast(F, L, m)["r_hat"].first_valid_index()
        if first is None:
            raise ValueError(f"No valid forecast for {F}, L={L}, m={m}")
        firsts.append(first)
    return max(firsts)


@dataclass
class GridResult:
    configs: pd.DataFrame          # index config_id
    dates: pd.DatetimeIndex
    exret: np.ndarray              # ExRet_GDX, aligned to dates
    pos: np.ndarray                # int8 [n_config, n_dates], pos after close t
    new_trade: np.ndarray          # int8 [n_config, n_dates]
    signal_start: pd.Timestamp     # first close on which any config may trade
    n_tested: int                  # size of the selectable grid (multiple testing)

    @property
    def eval_mask(self) -> np.ndarray:
        """P&L rows that are scored: strictly after signal_start."""
        return np.asarray(self.dates > self.signal_start)

    def net_returns(self, cost_bps: float, rows=None, mask: np.ndarray | None = None) -> pd.DataFrame:
        """dates x configs daily net returns: pos_{t-1} * r_t - cost * |dpos_{t-1}|.

        ``rows`` selects configs (labels or boolean); ``mask`` restricts dates
        (defaults to the common evaluation window).
        """
        sel = np.arange(len(self.configs)) if rows is None else \
            self.configs.index.get_indexer(self.configs.loc[rows].index)
        pos = self.pos[sel].astype(np.float32)
        r = np.nan_to_num(self.exret, nan=0.0).astype(np.float32)
        prev = np.concatenate([np.zeros((len(sel), 1), np.float32), pos[:, :-1]], axis=1)
        turn = np.abs(np.diff(np.concatenate([np.zeros((len(sel), 1), np.float32), pos], axis=1), axis=1))
        turn_prev = np.concatenate([np.zeros((len(sel), 1), np.float32), turn[:, :-1]], axis=1)
        net = prev * r[None, :] - turn_prev * np.float32(cost_bps * 1e-4)
        m = self.eval_mask if mask is None else mask
        return pd.DataFrame(net[:, m].T, index=self.dates[m], columns=self.configs.index[sel])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pos=self.pos, new_trade=self.new_trade, exret=self.exret,
                            dates=self.dates.values,
                            signal_start=np.int64(self.signal_start.value),
                            n_tested=self.n_tested)
        self.configs.assign(factors=self.configs["factors"].map("+".join)).to_csv(path.with_suffix(".configs.csv"))

    @classmethod
    def load(cls, path: str | Path) -> "GridResult":
        path = Path(path)
        z = np.load(path)
        cfg = pd.read_csv(path.with_suffix(".configs.csv"), index_col="config_id")
        cfg["factors"] = cfg["factors"].map(lambda s: tuple(s.split("+")))
        return cls(cfg, pd.DatetimeIndex(z["dates"], name="Date"), z["exret"], z["pos"], z["new_trade"],
                   pd.Timestamp(int(z["signal_start"])), int(z["n_tested"]))


def run_configs(returns: pd.DataFrame, configs: pd.DataFrame, fixed: dict,
                signal_start: pd.Timestamp, n_tested: int | None = None) -> GridResult:
    """Run an explicit set of configs (rows with GRID_KEYS columns), all starting flat
    with signals zeroed before ``signal_start``."""
    rc = ResidualCache(returns, min_obs_frac=fixed["stage1_min_obs_frac"])
    fcache = ForecastCache(rc, window=fixed["stage2_window"], min_obs=fixed["stage2_min_obs"])
    start = pd.Timestamp(signal_start)
    dates = returns.index
    before = np.asarray(dates < start)
    pos = np.zeros((len(configs), len(dates)), np.int8)
    new = np.zeros_like(pos)
    row_of = {cid: i for i, cid in enumerate(configs.index)}
    for (F, L, m), grp in configs.groupby(["factors", "lookback", "m"], sort=False):
        fc = fcache.forecast(F, L, m)
        for (k, dmode), g2 in grp.groupby(["k", "direction_mode"], sort=False):
            sig = build_signal(fc, k, dmode, fixed["min_edge_bps"]).to_numpy().copy()
            sig[before] = 0
            for cid, c in g2.iterrows():
                p, _, nt, _ = run_engine(sig, c["H"], c["policy"])
                pos[row_of[cid]], new[row_of[cid]] = p, nt
    return GridResult(configs, dates, returns["ExRet_GDX"].to_numpy(float), pos, new, start,
                      len(configs) if n_tested is None else n_tested)


def run_grid(returns: pd.DataFrame, cfg: dict, include_reference: bool = False,
             signal_start: pd.Timestamp | None = None) -> GridResult:
    """Run every config. ``signal_start`` overrides the computed common start
    (used for OOS, where warm-up comes from IS history)."""
    grid, fixed = cfg["grid"], cfg["fixed"]
    policies = list(grid["policy"])
    if include_reference:
        policies += cfg.get("reference_only", {}).get("policy", [])
    configs = enumerate_configs(grid, policies)
    if signal_start is None:
        rc = ResidualCache(returns, min_obs_frac=fixed["stage1_min_obs_frac"])
        fcache = ForecastCache(rc, window=fixed["stage2_window"], min_obs=fixed["stage2_min_obs"])
        signal_start = common_signal_start(fcache, grid)
    log.info("Grid: %d configs (%d selectable); signals from %s, scored from next row",
             len(configs), n_configs(grid), pd.Timestamp(signal_start).date())
    return run_configs(returns, configs, fixed, signal_start, n_configs(grid))
