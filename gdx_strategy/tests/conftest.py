import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IS_WORKBOOK = ROOT / "data" / "raw" / "gold_etf_factor_regression.xlsx"


@pytest.fixture(scope="session")
def is_workbook() -> Path:
    if not IS_WORKBOOK.exists():
        pytest.skip(f"IS workbook not found at {IS_WORKBOOK}")
    return IS_WORKBOOK
