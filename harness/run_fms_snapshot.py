"""
Offline FMS snapshot runner — Validation Harness (manual scenario).

Purpose
-------
Inject checked-in CSV fixtures into production scoring APIs
(``momentum_now_and_delta`` / ``compute_fms_snapshot``) without calling
yfinance or Finviz. Use this to eyeball FMS ranks and filter status before
or after formula changes.

Usage (from repo root)
----------------------
    python -m harness.run_fms_snapshot
    python -m harness.run_fms_snapshot --no-ohlc
    python -m harness.run_fms_snapshot --prices tests/fixtures/synthetic_prices_krw.csv

Related automated asserts: ``tests/unit/test_fms_scoring.py``.
Do not import this module from ``app.py`` or ``run_scan_batch.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_utils import momentum_now_and_delta

DEFAULT_PRICES = Path("tests/fixtures/synthetic_prices_krw.csv")
DEFAULT_OHLC = Path("tests/fixtures/synthetic_ohlc.csv")


def load_prices(path: Path) -> pd.DataFrame:
    """Load a KRW close-price panel CSV (Date index)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def load_ohlc(path: Path | None) -> pd.DataFrame | None:
    """Load MultiIndex OHLC CSV, or None if path is missing/disabled."""
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)


def run(prices_path: Path, ohlc_path: Path | None, top: int) -> pd.DataFrame:
    """Score injected fixtures and return the ranked FMS table."""
    prices = load_prices(prices_path)
    ohlc = load_ohlc(ohlc_path)
    symbols = list(prices.columns)
    return momentum_now_and_delta(
        prices,
        reference_prices_krw=prices,
        ohlc_data=ohlc,
        symbols=symbols,
    ).head(top)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the FMS fixture harness."""
    parser = argparse.ArgumentParser(
        description="Run FMS scoring on fixture CSVs (no market API)."
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--ohlc", type=Path, default=DEFAULT_OHLC)
    parser.add_argument(
        "--no-ohlc",
        action="store_true",
        help="Skip tradeability filter input",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    ohlc_path = None if args.no_ohlc else args.ohlc
    result = run(args.prices, ohlc_path, args.top)
    cols = [
        c
        for c in ["FMS", "ΔFMS_1D", "ΔFMS_5D", "Filter_Status", "R_1M", "R_3M"]
        if c in result.columns
    ]
    print(result[cols].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
