"""
Generate checked-in synthetic KRW price / OHLC fixtures for the FMS harness.

Purpose
-------
Reproduce ``tests/fixtures/synthetic_*.csv`` offline (no market API) with a
fixed RNG seed so golden ranks stay stable across machines.

Usage (from repo root)
----------------------
    python scripts/fixtures/generate_synthetic_panel.py
    python scripts/fixtures/generate_synthetic_panel.py --out-dir tests/fixtures

After regenerating, re-run::

    python -m pytest tests/unit/test_fms_scoring.py
    python -m harness.run_fms_snapshot

If ranks change intentionally, update ``tests/fixtures/golden_fms_ranks.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
SEED = 42
N_DAYS = 180


def build_prices(n: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Build a four-symbol synthetic KRW close panel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)

    def geom(start: float, daily_rets: np.ndarray) -> np.ndarray:
        return start * np.cumprod(1.0 + daily_rets)

    trend = geom(100.0, np.full(n, 0.004) + rng.normal(0, 0.002, n))
    mild = geom(100.0, np.full(n, 0.0015) + rng.normal(0, 0.005, n))
    flat = geom(100.0, rng.normal(0, 0.012, n))
    crash_rets = rng.normal(0, 0.015, n)
    crash_rets[-10] = -0.35
    crash_rets[-8] = -0.32
    crash_rets[-6] = -0.08
    crash_rets[-5] = -0.09
    crash_rets[-4] = -0.10
    crash_rets[-3] = -0.11
    crash = geom(100.0, crash_rets)

    prices = pd.DataFrame(
        {
            "TREND_UP": trend,
            "MILD_UP": mild,
            "FLAT": flat,
            "CRASHY": crash,
        },
        index=dates,
    )
    prices.index.name = "Date"
    return prices


def build_ohlc(prices: pd.DataFrame) -> pd.DataFrame:
    """Build MultiIndex OHLC so CRASHY trips tradeability disqualification."""
    frames = {}
    for sym in prices.columns:
        close = prices[sym]
        if sym != "CRASHY":
            high = close * 1.01
            low = close * 0.99
        else:
            high = close * 1.05
            low = close.copy()
            low.iloc[-10] = close.iloc[-11] * 0.60
            high.iloc[-10] = close.iloc[-11] * 1.02
            low.iloc[-8] = close.iloc[-9] * 0.65
            high.iloc[-8] = close.iloc[-9] * 1.02
            for i in (-6, -5, -4, -3):
                low.iloc[i] = close.iloc[i - 1] * 0.90
                high.iloc[i] = close.iloc[i - 1] * 1.01
        frames[sym] = pd.DataFrame({"High": high, "Low": low, "Close": close})
    return pd.concat(frames, axis=1)


def main(argv: list[str] | None = None) -> int:
    """Write synthetic price and OHLC CSVs under the fixtures directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prices = build_prices(seed=args.seed)
    ohlc = build_ohlc(prices)
    prices_path = args.out_dir / "synthetic_prices_krw.csv"
    ohlc_path = args.out_dir / "synthetic_ohlc.csv"
    prices.to_csv(prices_path)
    ohlc.to_csv(ohlc_path)
    print(f"Wrote {prices_path} shape={prices.shape}")
    print(f"Wrote {ohlc_path} shape={ohlc.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
