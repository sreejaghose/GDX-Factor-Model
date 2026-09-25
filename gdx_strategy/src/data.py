"""Load prices and Fed Funds from the workbook and build simple / excess returns.

Everything is recomputed in Python from the two input sheets; the workbook's
``Daily Data`` formulas are not used (they are only a validation target).

Timing convention: row ``t`` holds quantities known at the close of day ``t``.
``Ret_X`` on row ``t`` is the close(t-1) -> close(t) return.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRICES_SHEET = "Prices (input)"
FEDFUNDS_SHEET = "FedFunds (input)"
HEADER_ROW = 1  # 0-based: header is on Excel row 2
ASSETS = ["SPY", "IAU", "GLD", "TLT", "GDX"]  # GDXJ dropped
TRADING_DAYS = 252

RFMode = Literal["same_month", "prior_month"]


def load_prices(path: str | Path) -> pd.DataFrame:
    """Daily closes indexed by date; 0 / blank -> NaN; GDXJ dropped."""
    df = pd.read_excel(path, sheet_name=PRICES_SHEET, header=HEADER_ROW)
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    missing = [a for a in ASSETS if a not in df.columns]
    if missing:
        raise ValueError(f"Prices sheet missing columns: {missing}")
    px = df[ASSETS].apply(pd.to_numeric, errors="coerce")
    px = px.where(px > 0)  # 0 or negative prices are treated as missing
    if px.index.has_duplicates:
        raise ValueError("Duplicate dates in prices sheet")
    return px


def load_fedfunds(path: str | Path) -> pd.Series:
    """Monthly FEDFUNDS in percent, indexed by month-start date."""
    df = pd.read_excel(path, sheet_name=FEDFUNDS_SHEET, header=HEADER_ROW)
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.dropna(subset=["Date"])
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp(),
        name="FEDFUNDS",
    ).sort_index().dropna()
    return s


def daily_rf(dates: pd.DatetimeIndex, fedfunds: pd.Series, rf_mode: RFMode = "prior_month") -> pd.Series:
    """Annualised risk-free rate (decimal) for each trading date.

    same_month  : day in month m uses FEDFUNDS(m)   -- matches the workbook's LOOKUP
                  but FEDFUNDS(m) is a monthly average not known until m ends.
    prior_month : day in month m uses FEDFUNDS(m-1) -- known at the time.
    """
    if rf_mode not in ("same_month", "prior_month"):
        raise ValueError(f"rf_mode must be 'same_month' or 'prior_month', got {rf_mode!r}")
    ff = fedfunds / 100.0
    if rf_mode == "prior_month":
        # FEDFUNDS(m) becomes usable from the first day of month m+1
        ff = ff.copy()
        ff.index = ff.index + pd.offsets.MonthBegin(1)
    # as-of (backward) lookup, same semantics as Excel LOOKUP on sorted dates
    rf = ff.reindex(ff.index.union(dates)).ffill().reindex(dates)
    if rf.isna().any():
        raise ValueError(f"No FEDFUNDS value available for {int(rf.isna().sum())} dates "
                         f"(first: {rf[rf.isna()].index[0].date()})")
    rf.name = "RF_annual"
    return rf


def build_returns(prices: pd.DataFrame, fedfunds: pd.Series, rf_mode: RFMode = "prior_month") -> pd.DataFrame:
    """Columns: RF_annual, Ret_<X>, ExRet_<X>. First (return-less) row dropped."""
    rets = prices.pct_change(fill_method=None)  # NaN if either price is missing
    rf = daily_rf(prices.index, fedfunds, rf_mode)
    out = pd.DataFrame({"RF_annual": rf})
    for a in prices.columns:
        out[f"Ret_{a}"] = rets[a]
        out[f"ExRet_{a}"] = rets[a] - rf / TRADING_DAYS
    return out.iloc[1:]


def check_adjusted_closes(prices: pd.DataFrame, t_crit: float = -2.0) -> dict[str, dict]:
    """Heuristic test that SPY/TLT closes are dividend-adjusted.

    On unadjusted closes, ex-dividend days show a systematic price drop
    (~0.4-0.5% for SPY quarterly, ~0.2-0.3% for TLT monthly). We compare mean
    returns on approximate ex-dividend days with other days:
      SPY: first trading day on/after the 3rd Friday of Mar/Jun/Sep/Dec
      TLT: first trading day of each month
    A significantly negative difference -> warning that closes look unadjusted.
    """
    rets = prices.pct_change(fill_method=None).iloc[1:]
    idx = rets.index
    report: dict[str, dict] = {}

    # SPY: approx ex-div = first trading day >= 3rd Friday of quarter-end months
    spy_days = []
    for y in range(idx.year.min(), idx.year.max() + 1):
        for m in (3, 6, 9, 12):
            third_fri = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
            third_fri = third_fri[(third_fri.month == m) & (third_fri.weekday == 4)][2]
            pos = idx.searchsorted(third_fri)
            if pos < len(idx):
                spy_days.append(idx[pos])
    spy_mask = idx.isin(spy_days)

    # TLT: first trading day of each month
    month = idx.to_period("M")
    tlt_mask = np.r_[True, month[1:] != month[:-1]]

    for asset, mask, drop in (("SPY", spy_mask, 0.0045), ("TLT", tlt_mask, 0.0025)):
        r = rets[asset].dropna()
        m = pd.Series(mask, index=idx).reindex(r.index)
        on, off = r[m], r[~m]
        diff = on.mean() - off.mean()
        se = np.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))
        t = diff / se
        verdict = "looks unadjusted" if t < t_crit else "consistent with adjusted"
        report[asset] = dict(n_exdiv=int(len(on)), mean_diff_bps=diff * 1e4, t_stat=t,
                             expected_drop_if_unadjusted_bps=-drop * 1e4, verdict=verdict)
        msg = (f"{asset} ex-div-day return minus other days: {diff * 1e4:+.1f} bps "
               f"(t={t:+.2f}, n={len(on)}; unadjusted would be ~{-drop * 1e4:.0f} bps) -> {verdict}")
        if t < t_crit:
            log.warning(msg)
        else:
            log.info(msg)
    if any(v["verdict"] != "consistent with adjusted" for v in report.values()):
        log.warning("Prices may not be total-return/adjusted closes; factor betas and "
                    "excess returns would be biased on dividend dates.")
    return report


def load_dataset(path: str | Path, rf_mode: RFMode = "prior_month",
                 check_adjusted: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, returns) for the workbook at ``path``."""
    prices = load_prices(path)
    fedfunds = load_fedfunds(path)
    if check_adjusted:
        check_adjusted_closes(prices)
    returns = build_returns(prices, fedfunds, rf_mode)
    return prices, returns


class SeamError(ValueError):
    """IS and OOS data do not join cleanly."""


def load_is_oos(is_path: str | Path, oos_path: str | Path, rf_mode: RFMode = "prior_month",
                max_gap_weekdays: int = 4, price_rtol: float = 1e-6) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Concatenate IS and OOS prices, check the seam, and build returns.

    * Overlapping dates must carry identical prices (within ``price_rtol``); they are
      then dropped from OOS. Differing prices, or OOS dates inside the IS range that
      IS doesn't have, raise SeamError.
    * More than ``max_gap_weekdays`` weekdays strictly between the last IS date and
      the first OOS date raises SeamError (the first OOS return would span the gap).
    * FEDFUNDS: IS values are kept for IS months (so IS returns reproduce exactly);
      OOS values fill later months.
    """
    px_is, px_oos = load_prices(is_path), load_prices(oos_path)
    if px_oos.empty:
        raise SeamError("OOS file has no price rows")
    is_last = px_is.index[-1]
    info: dict = {"is_first": px_is.index[0], "is_last": is_last, "n_is_rows": len(px_is)}
    inside = px_oos.index[px_oos.index <= is_last]
    if len(inside):
        missing = inside.difference(px_is.index)
        if len(missing):
            raise SeamError(f"OOS has {len(missing)} dates inside the IS range that IS lacks "
                            f"(first {missing[0].date()})")
        a, b = px_is.loc[inside], px_oos.loc[inside]
        both = a.notna() & b.notna()
        diff = ((a - b).abs() / a.abs()).where(both)
        if (diff > price_rtol).any().any():
            raise SeamError(f"OOS overlaps IS on {len(inside)} dates with different prices "
                            f"(max rel diff {float(np.nanmax(diff.to_numpy())):.2e})")
        log.warning("OOS overlaps IS on %d identical dates (%s -> %s); dropped from OOS",
                    len(inside), inside[0].date(), inside[-1].date())
        px_oos = px_oos.loc[px_oos.index > is_last]
        if px_oos.empty:
            raise SeamError("OOS file contains no dates after the IS period")
    gap = int(np.busday_count((is_last + pd.Timedelta(days=1)).date(), px_oos.index[0].date()))
    if gap > max_gap_weekdays:
        raise SeamError(f"{gap} weekdays missing between IS end {is_last.date()} and OOS start "
                        f"{px_oos.index[0].date()} (max {max_gap_weekdays})")
    ff = load_fedfunds(is_path).combine_first(load_fedfunds(oos_path))
    prices = pd.concat([px_is, px_oos])
    returns = build_returns(prices, ff, rf_mode)
    info.update(oos_first=px_oos.index[0], oos_last=px_oos.index[-1], n_oos_rows=len(px_oos),
                overlap_dropped=len(inside), weekdays_between=gap,
                oos_missing_prices=int(px_oos.isna().sum().sum()),
                first_oos_return_gdx=float(returns.loc[px_oos.index[0], "Ret_GDX"]))
    return prices, returns, info
