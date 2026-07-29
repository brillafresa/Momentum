"""Plot largest development-set residual symbols from the locked snapshot."""

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


def main() -> None:
    manifest = load_manifest()
    residuals = pd.read_csv(RESIDUAL_CSV)
    scores = pd.read_csv(SCORES_CSV, index_col=0)
    prices = pd.read_pickle(
        os.path.join(
            SNAPSHOT_ROOT_DIR, manifest.snapshot_id, "prices_krw.pkl"
        )
    )

    symbols = []
    for row in residuals.itertuples(index=False):
        for symbol in (row.left, row.right):
            if symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) >= 8:
                break
        if len(symbols) >= 8:
            break

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    for ax, symbol in zip(axes.flat, symbols):
        series = prices[symbol].dropna().iloc[-63:]
        rebased = series / series.iloc[0] * 100.0
        ema20 = ema(series, 20) / series.iloc[0] * 100.0
        ax.plot(rebased.index, rebased, label="Price", linewidth=1.8)
        ax.plot(ema20.index, ema20, label="EMA20", linewidth=1.2)
        row = scores.loc[symbol]
        ax.set_title(
            f"{symbol}: true {int(row.true_rank)}, "
            f"scratch {int(row.scratch_rank)}"
        )
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
