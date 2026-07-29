"""
Offline old/new FMS contribution comparison for the cash-like gate.

Usage (from repo root)::

    python -m harness.compare_cash_like_gate
    python -m harness.compare_cash_like_gate --prices fms_calibration_snapshots/fms_20260729_154752/prices_krw.pkl
    python -m harness.compare_cash_like_gate --symbols 449170.KS 459580.KS 499660.KS

No live network. Inject a price panel (fixture CSV or local pickle).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from core.fms_features import (
    PRODUCTION_FMS_COLUMNS,
    build_panel_feature_frame,
    cash_like_strength,
    production_axis_contributions,
    score_production_fms_features,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "tests" / "fixtures" / "cash_like_paths_prices_krw.csv"


def load_prices(path: Path) -> pd.DataFrame:
    """Load a Date-indexed price panel from CSV or pickle."""
    if path.suffix.lower() == ".pkl":
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"expected DataFrame pickle, got {type(obj)}")
        return obj
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def compare_panel(
    prices: pd.DataFrame,
    *,
    symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return per-symbol old/new FMS, cash strength, and gated-axis deltas."""
    cols = list(symbols) if symbols is not None else list(prices.columns)
    cols = [c for c in cols if c in prices.columns]
    feats = build_panel_feature_frame(prices, symbols=cols)
    strength = cash_like_strength(feats)
    old = score_production_fms_features(feats, apply_cash_gate=False)
    new = score_production_fms_features(feats, apply_cash_gate=True)
    old_c = production_axis_contributions(feats, apply_cash_gate=False)
    new_c = production_axis_contributions(feats, apply_cash_gate=True)

    out = pd.DataFrame(
        {
            "cash_strength": strength,
            "FMS_old": old,
            "FMS_new": new,
            "FMS_delta": new - old,
            "R_3M": feats["R_3M"],
            "R2_3M": feats["R2_3M"],
            "Vol20_Ann": feats["Vol20_Ann"] if "Vol20_Ann" in feats.columns else float("nan"),
        },
        index=feats.index,
    )
    for col in PRODUCTION_FMS_COLUMNS:
        out[f"d_{col}"] = new_c[col] - old_c[col]
    return out.sort_values("FMS_delta")


def summarize(delta: pd.Series) -> dict:
    """Compact distribution summary for impact reporting."""
    finite = delta.dropna()
    if finite.empty:
        return {}
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "p10": float(finite.quantile(0.10)),
        "p90": float(finite.quantile(0.90)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n_changed": int((finite.abs() > 1e-12).sum()),
        "n_unchanged": int((finite.abs() <= 1e-12).sum()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    args = parser.parse_args(argv)

    prices = load_prices(args.prices)
    report = compare_panel(prices, symbols=args.symbols)
    print(f"panel={args.prices} symbols={len(report)}")
    print("delta_summary=", summarize(report["FMS_delta"]))
    gated = report[report["cash_strength"] > 0.05].sort_values("cash_strength", ascending=False)
    print(f"cash_strength>0.05 count={len(gated)}")
    if not gated.empty:
        print(gated.head(args.top).to_string(float_format=lambda x: f"{x: .4f}"))
    else:
        print(report.head(args.top).to_string(float_format=lambda x: f"{x: .4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
