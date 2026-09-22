"""IS protocol: development / validation masks, a logged validation gate, and an
anchored walk-forward check of the selection rule.

A *selection rule* is any callable ``rule(grid, mask) -> config_id`` that picks a
config using only the P&L rows in ``mask``. The same rule is used for the
development shortlist and for every walk-forward year.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from src.grid import GridResult

log = logging.getLogger(__name__)

SelectionRule = Callable[[GridResult, np.ndarray], str]
PARAM_COLS = ["factors", "lookback", "m", "k", "H", "policy", "direction_mode"]


def load_splits(path: str | Path) -> dict:
    with open(path) as f:
        s = yaml.safe_load(f)
    for name in ("development", "validation"):
        for key in ("start", "end"):
            v = s[name][key]
            s[name][key] = v if v == "auto" else pd.Timestamp(v)
    if s["development"]["end"] >= s["validation"]["start"]:
        raise ValueError("development must end before validation starts")
    log_path = Path(s["validation_log"])
    if not log_path.is_absolute():  # relative to the project root (parent of config/)
        s["validation_log"] = Path(path).resolve().parents[1] / log_path
    return s


def split_masks(grid: GridResult, splits: dict) -> dict[str, np.ndarray]:
    """Boolean masks over grid.dates for the scored rows of each split."""
    d = grid.dates
    ev = grid.eval_mask
    dev_start = splits["development"]["start"]
    dev = ev & np.asarray(d <= splits["development"]["end"])
    if dev_start != "auto":
        dev &= np.asarray(d >= dev_start)
    val = ev & np.asarray((d >= splits["validation"]["start"]) & (d <= splits["validation"]["end"]))
    return {"development": dev, "validation": val}


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def _shortlist_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


class ValidationGate:
    """Append-only log of validation evaluations.

    The first evaluation is free. Evaluating a *different* shortlist afterwards
    (i.e. iterating after seeing validation) raises unless a ``reason`` is given,
    in which case it is recorded as a protocol deviation.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def check_and_log(self, shortlist: list[str], reason: str | None = None) -> dict:
        h = _shortlist_hash(shortlist)
        prior = self.entries()
        seen = {e["shortlist_hash"] for e in prior}
        deviation = bool(prior) and h not in seen
        if deviation and not reason:
            raise PermissionError(
                "Validation was already evaluated for a different shortlist "
                f"({len(prior)} prior entr{'y' if len(prior) == 1 else 'ies'}). Re-selecting after "
                "seeing validation results is a protocol deviation: pass a reason to log it.")
        entry = dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     shortlist_hash=h, n_configs=len(shortlist), shortlist=sorted(shortlist),
                     evaluation_number=len(prior) + 1, deviation=deviation, reason=reason)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if deviation:
            log.warning("PROTOCOL DEVIATION logged: validation re-run on a new shortlist (%s)", reason)
        return entry


def evaluate_validation(grid: GridResult, shortlist: list[str], splits: dict, cost_bps: float,
                        reason: str | None = None) -> pd.DataFrame:
    """Daily net returns of the shortlist over validation. Logged through the gate."""
    missing = [c for c in shortlist if c not in grid.configs.index]
    if missing:
        raise KeyError(f"Unknown config ids: {missing}")
    ValidationGate(splits["validation_log"]).check_and_log(shortlist, reason)
    return grid.net_returns(cost_bps, rows=shortlist, mask=split_masks(grid, splits)["validation"])


# ---------------------------------------------------------------------------
# Anchored walk-forward
# ---------------------------------------------------------------------------

def _sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(252)) if s > 0 else np.nan


def walk_forward(grid: GridResult, rule: SelectionRule, first_year: int, last_year: int,
                 cost_bps: float, train_start: pd.Timestamp | None = None) -> dict:
    """For each Y: pick with ``rule`` on scored rows in [train_start, Y-1], trade Y.

    The chosen config's P&L in Y is its continuous-run P&L (signals are purely
    rolling, so it is what that config would have produced). When the config
    changes, the trade at the last close of Y-1 is re-costed as old position ->
    new position (``switch_cost`` is the adjustment and can be negative).

    Returns {'years': per-year table, 'returns': stitched daily net returns,
             'stability': summary of how much the choice moves}.
    """
    d = grid.dates
    ev = grid.eval_mask
    if train_start is not None:
        ev = ev & np.asarray(d >= train_start)
    years = pd.Index(d.year)
    rows, pieces = [], []
    prev_id = None
    for Y in range(first_year, last_year + 1):
        train = ev & np.asarray(years < Y)
        test = ev & np.asarray(years == Y)
        if not train.any() or not test.any():
            continue
        cid = rule(grid, train)
        r = grid.net_returns(cost_bps, rows=[cid], mask=test).iloc[:, 0]
        switch_cost = 0.0
        if prev_id is not None and cid != prev_id:
            # At the last close before Y we actually trade from the old config's
            # position (held into that close) to the new config's position; the new
            # config's own first-row cost assumed it had been running, so swap it out.
            j = np.flatnonzero(train)[-1]  # last close before Y
            i_new, i_old = grid.configs.index.get_indexer([cid, prev_id])
            new_j, new_prev, old_prev = (int(grid.pos[i_new, j]), int(grid.pos[i_new, j - 1]),
                                         int(grid.pos[i_old, j - 1]))
            switch_cost = (abs(new_j - old_prev) - abs(new_j - new_prev)) * cost_bps * 1e-4
            r.iloc[0] -= switch_cost
        train_r = grid.net_returns(cost_bps, rows=[cid], mask=train).iloc[:, 0].to_numpy()
        c = grid.configs.loc[cid]
        rows.append(dict(year=Y, config_id=cid, **{p: c[p] for p in PARAM_COLS},
                         train_sharpe=_sharpe(train_r), test_sharpe=_sharpe(r.to_numpy()),
                         test_return=float(r.sum()), switch_cost=switch_cost,
                         n_new_trades=int(grid.new_trade[grid.configs.index.get_loc(cid)][test].sum())))
        pieces.append(r)
        prev_id = cid
    tab = pd.DataFrame(rows).set_index("year")
    stitched = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return {"years": tab, "returns": stitched, "stability": selection_stability(tab)}


def selection_stability(tab: pd.DataFrame) -> dict:
    """How much the walk-forward choice moves from year to year."""
    if tab.empty:
        return {}
    n = len(tab)
    changes = [int(sum(a != b for a, b in zip(tab.iloc[i][PARAM_COLS], tab.iloc[i - 1][PARAM_COLS])))
               for i in range(1, n)]
    modal_share = {p: float(tab[p].astype(str).value_counts(normalize=True).iloc[0]) for p in PARAM_COLS}
    return dict(
        n_years=n,
        n_distinct_configs=int(tab["config_id"].nunique()),
        n_config_switches=int((tab["config_id"] != tab["config_id"].shift()).iloc[1:].sum()),
        mean_params_changed_per_switch=float(np.mean(changes)) if changes else 0.0,
        modal_share=modal_share,
        # heuristic flag: a new config most years, or >2 of 7 params moving on average
        red_flag=bool(tab["config_id"].nunique() > n / 2 or (changes and np.mean(changes) > 2)),
    )
