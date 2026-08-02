"""Plot largest residual symbols from the scratch/candidate residual CSV.

Purpose
-------
Offline visual review of symbols where the candidate (or promoted) score
rank diverges most from the ground-truth ranking. Uses the locked snapshot
price panel — no live market API.

Usage (from repo root)
----------------------
    python -m calibration.fms_recalib_plot_residuals

Inputs
------
- ``fms_recalib_scratch_residual_pairs.csv``
  Supported schemas:
  1) symbol-gap (nonlinear MC): ``symbol,true_rank,model_rank,gap,...``
  2) legacy pairwise (sparse refit): ``left,right,...``
- ``fms_recalib_scratch_scores.csv`` with ``rank`` / ``true_rank`` and
  ``candidate_score`` (optional ``scratch_rank`` / ``model_rank``)
- Snapshot prices under ``fms_price_snapshots/<snapshot_id>/prices_krw.pkl``

Output
------
``fms_recalib_scratch_residual_charts.png``
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from calibration.manifest import load_manifest
from calibration.session import SNAPSHOT_ROOT_DIR
from core.indicators import ema


RESIDUAL_CSV = "fms_recalib_scratch_residual_pairs.csv"
SCORES_CSV = "fms_recalib_scratch_scores.csv"
OUT_PNG = "fms_recalib_scratch_residual_charts.png"


def _symbols_from_residuals(residuals: pd.DataFrame, limit: int = 8) -> list[str]:
    symbols: list[str] = []
    if "symbol" in residuals.columns:
        ordered = residuals.copy()
        if "gap" in ordered.columns:
            ordered = ordered.reindex(
                ordered["gap"].abs().sort_values(ascending=False).index
            )
        for sym in ordered["symbol"].astype(str):
            if sym not in symbols:
                symbols.append(sym)
            if len(symbols) >= limit:
                break
        return symbols

    for row in residuals.itertuples(index=False):
        for symbol in (getattr(row, "left", None), getattr(row, "right", None)):
            if symbol is None or str(symbol) in symbols:
                continue
            symbols.append(str(symbol))
            if len(symbols) >= limit:
                return symbols
    return symbols


def _title_ranks(scores: pd.DataFrame, symbol: str) -> str:
    row = scores.loc[symbol]
    true_rank = row["true_rank"] if "true_rank" in scores.columns else row.get("rank")
    if "scratch_rank" in scores.columns:
        model_rank = row["scratch_rank"]
    elif "model_rank" in scores.columns:
        model_rank = row["model_rank"]
    elif "candidate_score" in scores.columns:
        model_rank = scores["candidate_score"].rank(ascending=False, method="average")[
            symbol
        ]
    else:
        model_rank = float("nan")
    return f"{symbol}: true {int(true_rank)}, model {float(model_rank):.0f}"


def main() -> None:
    manifest = load_manifest()
    residuals = pd.read_csv(RESIDUAL_CSV)
    scores = pd.read_csv(SCORES_CSV, index_col=0)
    prices = pd.read_pickle(
        os.path.join(SNAPSHOT_ROOT_DIR, manifest.snapshot_id, "prices_krw.pkl")
    )

    symbols = [
        s for s in _symbols_from_residuals(residuals, limit=8) if s in prices.columns
    ]
    if not symbols:
        raise RuntimeError(f"No plottable symbols found in {RESIDUAL_CSV}")

    n = len(symbols)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.5 * nrows), constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, symbol in zip(axes_flat, symbols):
        series = prices[symbol].dropna().iloc[-63:]
        rebased = series / series.iloc[0] * 100.0
        ema20 = ema(series, 20) / series.iloc[0] * 100.0
        ax.plot(rebased.index, rebased, label="Price", linewidth=1.8)
        ax.plot(ema20.index, ema20, label="EMA20", linewidth=1.2)
        ax.set_title(_title_ranks(scores, symbol))
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    for ax in axes_flat[len(symbols) :]:
        ax.set_visible(False)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
