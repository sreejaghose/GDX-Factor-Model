"""The Python pipeline must reproduce the workbook's Full-Sample Sweep for GDX.

Uses rf_mode='same_month' (the workbook's Fed Funds mapping). If this fails,
fix the data pipeline before anything downstream is trusted.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import load_dataset
from src.factor_model import factor_subsets, full_sample_sweep

TOL = 1e-4
SWEEP_SHEET = "Full-Sample Sweep"
SWEEP_HEADER_ROW = 3  # 0-based: header on Excel row 4
COMPARE = ["N", "Alpha_bps", "Alpha_t", "Alpha_p", "R2", "AdjR2", "AIC", "BIC"] + [
    f"{s}_{f}" for f in ("SPY", "IAU", "GLD", "TLT") for s in ("Beta", "t", "p")
]


@pytest.fixture(scope="module")
def excel_sweep(is_workbook) -> pd.DataFrame:
    s = pd.read_excel(is_workbook, sheet_name=SWEEP_SHEET, header=SWEEP_HEADER_ROW)
    return s[s["Vehicle"] == "GDX"].set_index("Factors")


@pytest.fixture(scope="module")
def python_sweep(is_workbook) -> pd.DataFrame:
    _, returns = load_dataset(is_workbook, rf_mode="same_month", check_adjusted=False)
    return full_sample_sweep(returns, target="GDX").set_index("Factors")


def test_all_subsets_present(excel_sweep):
    expected = {"+".join(s) for s in factor_subsets()}
    assert set(excel_sweep.index) == expected
    assert len(excel_sweep) == 15


@pytest.mark.parametrize("subset", ["+".join(s) for s in factor_subsets()])
def test_subset_matches_excel(subset, excel_sweep, python_sweep):
    xl, py = excel_sweep.loc[subset], python_sweep.loc[subset]
    for col in COMPARE:
        x = xl[col]
        if pd.isna(x):  # factor not in this subset
            assert col not in py.index or pd.isna(py[col]), f"{subset}: {col} unexpected"
            continue
        assert abs(py[col] - x) <= TOL, f"{subset}: {col} python={py[col]!r} excel={x!r}"


def test_headline_numbers(python_sweep):
    """Key facts quoted in the spec."""
    r = python_sweep.loc["SPY+GLD"]
    assert r["N"] == 3931
    assert r["Alpha_bps"] == pytest.approx(-3.93, abs=5e-3)
    assert r["Beta_SPY"] == pytest.approx(0.498, abs=5e-4)
    assert r["Beta_GLD"] == pytest.approx(1.744, abs=5e-4)
    assert r["R2"] == pytest.approx(0.6308, abs=5e-5)
    assert r["BIC"] == pytest.approx(-21304.2, abs=0.05)
    assert python_sweep["BIC"].idxmin() == "SPY+GLD"
    assert python_sweep["AIC"].idxmin() == "SPY+GLD+TLT"
    t = python_sweep.loc["SPY+GLD+TLT"]
    assert t["Beta_TLT"] == pytest.approx(0.085, abs=1e-3)
    assert t["t_TLT"] == pytest.approx(2.7, abs=0.05)
    # IAU is redundant alongside GLD
    assert abs(python_sweep.loc["SPY+IAU+GLD", "t_IAU"]) < 1.5
