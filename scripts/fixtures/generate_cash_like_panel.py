"""
Generate checked-in cash-like / bond / equity path fixtures for the FMS harness.

Purpose
-------
Reproduce ``tests/fixtures/cash_like_paths_prices_krw.csv`` offline (no market
API) so the cash-like quality-bonus gate and relative-Z regressions stay
reproducible.

Paths covered
-------------
- ``CASH_LIKE`` / ``CASH_STAIR`` — low return × ultra-low vol × high R² (gate on)
- ``NOISY_LOW`` — low return but equity-like noise (gate off)
- ``BOND_RALLY`` / ``EQUITY_TREND`` / ``SMOOTH_STRONG`` — quality paths (gate off)

Usage (from repo root)
----------------------
    python scripts/fixtures/generate_cash_like_panel.py
    python scripts/fixtures/generate_cash_like_panel.py --out-dir tests/fixtures

After regenerating, re-run::

    python -m pytest tests/unit/test_fms_cash_like_gate.py
    python -m harness.compare_cash_like_gate

Production note: ``app.py`` / ``run_scan_batch.py`` must not import this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
N_DAYS = 120
SEED = 7


def _geom(start: float, daily: np.ndarray) -> np.ndarray:
    return start * np.cumprod(1.0 + daily)


def build_cash_like_panel(n: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Build paths covering cash-gate contracts (no network)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)

    # ~3% annual cash rate: smooth low return, ultra-low vol, near-perfect R².
    cash_daily = np.full(n, 0.03 / 252.0)
    cash_like = _geom(100_000.0, cash_daily)

    # Stair-step cash rate (tiny discrete ticks, still ultra-low vol).
    stair = np.full(n, 100_000.0)
    for i in range(1, n):
        if i % 5 == 0:
            stair[i] = stair[i - 1] * (1.0 + 0.03 / 50.0)
        else:
            stair[i] = stair[i - 1]

    # Low 3M return but equity-like noise (should NOT trigger cash gate).
    noisy_low = _geom(10_000.0, rng.normal(0.00005, 0.015, n))

    # Strong long-bond style rally (~25% over ~3M window at end).
    bond_daily = np.concatenate(
        [rng.normal(0.0002, 0.004, n - 70), np.full(70, 0.0032)]
    )
    bond_rally = _geom(50_000.0, bond_daily)

    # Normal equity uptrend (should keep quality bonuses).
    equity = _geom(100.0, np.full(n, 0.004) + rng.normal(0, 0.008, n))

    # Smooth but high return (quality should remain credited).
    smooth_strong = _geom(100.0, np.full(n, 0.004))

    prices = pd.DataFrame(
        {
            "CASH_LIKE": cash_like,
            "CASH_STAIR": stair,
            "NOISY_LOW": noisy_low,
            "BOND_RALLY": bond_rally,
            "EQUITY_TREND": equity,
            "SMOOTH_STRONG": smooth_strong,
        },
        index=dates,
    )
    prices.index.name = "Date"
    return prices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for cash_like_paths_prices_krw.csv",
    )
    args = parser.parse_args(argv)
    out = args.out_dir / "cash_like_paths_prices_krw.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_cash_like_panel()
    panel.to_csv(out)
    print(f"wrote {out} shape={panel.shape}")


if __name__ == "__main__":
    main()
