"""Shared fixtures for the FMS test harness (no live market API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the checked-in fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def synthetic_prices_krw(fixtures_dir: Path) -> pd.DataFrame:
    """Load synthetic KRW close prices (Date index, one column per symbol)."""
    path = fixtures_dir / "synthetic_prices_krw.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="session")
def synthetic_ohlc(fixtures_dir: Path) -> pd.DataFrame:
    """Load synthetic OHLC with MultiIndex columns (symbol, field)."""
    path = fixtures_dir / "synthetic_ohlc.csv"
    return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)


@pytest.fixture(scope="session")
def golden_fms_ranks(fixtures_dir: Path) -> Dict[str, Any]:
    """Load golden FMS rank expectations for the synthetic panel."""
    with open(fixtures_dir / "golden_fms_ranks.json", encoding="utf-8") as f:
        return json.load(f)
