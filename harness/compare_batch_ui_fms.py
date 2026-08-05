# -*- coding: utf-8 -*-
"""
Compare FMS under batch vs UI calendar/panel paths (Validation Harness).

Purpose
-------
v5.0.0 production FMS is absolute (``alive_pullback``): reference panels do not
change scores. Residual batch↔UI differences come from how KRW price panels are
built (per-market ``align_bday_ffill`` + ``coverage=0.5`` in the UI vs shared
index + ``coverage=0.9`` in batch chunks).

This harness downloads **once** (or injects fixtures), then builds both panels
and scores with the same OHLC so wall-clock drift is eliminated. Use it to
quantify path-only dFMS. Empirically, after ffill the shared-symbol scores stay
nearly identical (rank correlation ~1); larger live gaps are usually timing.

Usage (from repo root)
----------------------
    # Offline: synthetic fixture with injected calendar gaps
    python -m harness.compare_batch_ui_fms --offline

    # LIVE: current FREE watchlist, same download → dual path
    python -m harness.compare_batch_ui_fms --live
    python -m harness.compare_batch_ui_fms --live --mode IRP --limit 40
    python -m harness.compare_batch_ui_fms --live --symbols AAPL 005930.KS 0700.HK
    python -m harness.compare_batch_ui_fms --live --mirror-io --limit 40

Do not import this module from ``app.py`` or ``run_scan_batch.py``.
Offline path is CI-safe; ``--live`` is manual only (market API).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_utils import (  # noqa: E402
    align_bday_ffill,
    classify,
    download_fx,
    download_ohlc_prices,
    download_prices,
    harmonize_calendar,
    momentum_now_and_delta,
)

DEFAULT_PRICES = ROOT / "tests" / "fixtures" / "synthetic_prices_krw.csv"
DEFAULT_OHLC = ROOT / "tests" / "fixtures" / "synthetic_ohlc.csv"


@dataclass(frozen=True)
class PathCompareResult:
    """Side-by-side FMS comparison for symbols present in both panels."""

    comparison: pd.DataFrame
    ui_only: list[str]
    batch_only: list[str]
    ui_n: int
    batch_n: int


def apply_fx_to_local_prices(
    local_prices: pd.DataFrame,
    *,
    usdkrw: pd.Series,
    jpykrw: pd.Series,
    hkdkrw: pd.Series,
) -> pd.DataFrame:
    """Convert a local-currency Adj Close panel to KRW (shared calendar index).

    Mirrors the batch chunk FX step: multiply in place on the shared index
    without per-market ``align_bday_ffill``.
    """
    if local_prices.empty:
        return local_prices.copy()
    out = local_prices.copy()
    usd_cols = [c for c in out.columns if classify(str(c)) == "USA"]
    jpy_cols = [c for c in out.columns if classify(str(c)) == "JPN"]
    hkg_cols = [c for c in out.columns if classify(str(c)) == "HKG"]
    if usd_cols and usdkrw is not None and not usdkrw.empty:
        fx = usdkrw.reindex(out.index).ffill()
        out[usd_cols] = out[usd_cols].mul(fx, axis=0)
    if jpy_cols and jpykrw is not None and not jpykrw.empty:
        fx = jpykrw.reindex(out.index).ffill()
        out[jpy_cols] = out[jpy_cols].mul(fx, axis=0)
    if hkg_cols and hkdkrw is not None and not hkdkrw.empty:
        fx = hkdkrw.reindex(out.index).ffill()
        out[hkg_cols] = out[hkg_cols].mul(fx, axis=0)
    return out


def build_batch_style_prices_krw(prices_krw_raw: pd.DataFrame) -> pd.DataFrame:
    """Batch chunk panel path: ``harmonize_calendar(..., coverage=0.9)``."""
    if prices_krw_raw.empty:
        return prices_krw_raw.copy()
    return harmonize_calendar(prices_krw_raw, coverage=0.9)


def build_ui_style_prices_krw(
    local_prices: pd.DataFrame,
    *,
    usdkrw: pd.Series,
    jpykrw: pd.Series,
    hkdkrw: pd.Series,
) -> pd.DataFrame:
    """UI ``build_prices_krw`` panel path (per-market align + coverage=0.5).

    Expects *local* Adj Close (pre-FX). KRW names are left as-is.
    """
    if local_prices.empty:
        return local_prices.copy()

    usd_cols = [c for c in local_prices.columns if classify(str(c)) == "USA"]
    kor_cols = [c for c in local_prices.columns if classify(str(c)) == "KOR"]
    jpy_cols = [c for c in local_prices.columns if classify(str(c)) == "JPN"]
    hkg_cols = [c for c in local_prices.columns if classify(str(c)) == "HKG"]

    usd_df = align_bday_ffill(local_prices[usd_cols]) if usd_cols else pd.DataFrame()
    kor_df = align_bday_ffill(local_prices[kor_cols]) if kor_cols else pd.DataFrame()
    jpy_df = align_bday_ffill(local_prices[jpy_cols]) if jpy_cols else pd.DataFrame()
    hkg_df = align_bday_ffill(local_prices[hkg_cols]) if hkg_cols else pd.DataFrame()

    usdkrw_a = (
        align_bday_ffill(usdkrw.to_frame()).iloc[:, 0]
        if usdkrw is not None and not usdkrw.empty
        else usdkrw
    )
    jpykrw_a = (
        align_bday_ffill(jpykrw.to_frame()).iloc[:, 0]
        if jpykrw is not None and not jpykrw.empty
        else jpykrw
    )
    hkdkrw_a = (
        align_bday_ffill(hkdkrw.to_frame()).iloc[:, 0]
        if hkdkrw is not None and not hkdkrw.empty
        else hkdkrw
    )

    frames: list[pd.DataFrame] = []
    if not usd_df.empty and usdkrw_a is not None and not usdkrw_a.empty:
        fx = usdkrw_a.reindex(usd_df.index).ffill()
        frames.append(usd_df.mul(fx, axis=0))
    if not kor_df.empty:
        frames.append(kor_df)
    if not jpy_df.empty and jpykrw_a is not None and not jpykrw_a.empty:
        fx = jpykrw_a.reindex(jpy_df.index).ffill()
        frames.append(jpy_df.mul(fx, axis=0))
    if not hkg_df.empty and hkdkrw_a is not None and not hkdkrw_a.empty:
        fx = hkdkrw_a.reindex(hkg_df.index).ffill()
        frames.append(hkg_df.mul(fx, axis=0))
    if not frames:
        return pd.DataFrame()

    prices_krw = pd.concat(frames, axis=1).sort_index()
    prices_krw = prices_krw.loc[:, ~prices_krw.columns.duplicated()]
    return harmonize_calendar(prices_krw, coverage=0.5)


def compare_fms_paths(
    ui_prices: pd.DataFrame,
    batch_prices: pd.DataFrame,
    ohlc_data: Optional[pd.DataFrame],
) -> PathCompareResult:
    """Score UI vs batch panels and return a ΔFMS table for the intersection."""
    ui_syms = [c for c in ui_prices.columns]
    batch_syms = [c for c in batch_prices.columns]
    common = sorted(set(ui_syms) & set(batch_syms))
    ui_only = sorted(set(ui_syms) - set(batch_syms))
    batch_only = sorted(set(batch_syms) - set(ui_syms))

    if not common:
        return PathCompareResult(
            comparison=pd.DataFrame(),
            ui_only=ui_only,
            batch_only=batch_only,
            ui_n=len(ui_syms),
            batch_n=len(batch_syms),
        )

    ui_mom = momentum_now_and_delta(
        ui_prices[common],
        reference_prices_krw=ui_prices[common],
        ohlc_data=ohlc_data,
        symbols=common,
    )
    batch_mom = momentum_now_and_delta(
        batch_prices[common],
        reference_prices_krw=batch_prices[common],
        ohlc_data=ohlc_data,
        symbols=common,
    )

    cmp = pd.DataFrame(
        {
            "FMS_UI": ui_mom["FMS"],
            "FMS_Batch": batch_mom["FMS"],
        }
    )
    cmp["dFMS_Batch_minus_UI"] = cmp["FMS_Batch"] - cmp["FMS_UI"]
    cmp["abs_d"] = cmp["dFMS_Batch_minus_UI"].abs()
    if "Filter_Status" in ui_mom.columns:
        cmp["Filter_UI"] = ui_mom["Filter_Status"]
    if "Filter_Status" in batch_mom.columns:
        cmp["Filter_Batch"] = batch_mom["Filter_Status"]
    cmp = cmp.sort_values("abs_d", ascending=False)
    return PathCompareResult(
        comparison=cmp,
        ui_only=ui_only,
        batch_only=batch_only,
        ui_n=len(ui_syms),
        batch_n=len(batch_syms),
    )


def summarize_comparison(result: PathCompareResult) -> str:
    """Human-readable summary of path ΔFMS."""
    lines: list[str] = []
    lines.append(
        f"Panels: UI={result.ui_n} symbols, Batch={result.batch_n} symbols, "
        f"common={len(result.comparison)}"
    )
    if result.ui_only:
        lines.append(f"UI-only (dropped by batch coverage?): {result.ui_only[:12]}")
    if result.batch_only:
        lines.append(f"Batch-only: {result.batch_only[:12]}")

    cmp = result.comparison
    if cmp.empty:
        lines.append("No overlapping symbols to compare.")
        return "\n".join(lines)

    # Keep -999 rows separately from finite-score stats
    dq = cmp[(cmp["FMS_UI"] == -999.0) | (cmp["FMS_Batch"] == -999.0)]
    both_finite = cmp[(cmp["FMS_UI"] != -999.0) & (cmp["FMS_Batch"] != -999.0)]

    lines.append(f"Both finite FMS: {len(both_finite)} / common {len(cmp)}")
    if not dq.empty:
        flip = dq[dq["FMS_UI"] != dq["FMS_Batch"]]
        lines.append(
            f"Tradeability -999 involved: {len(dq)} "
            f"(path disagreement on -999: {len(flip)})"
        )

    if both_finite.empty:
        lines.append("No finite-FMS pairs.")
        return "\n".join(lines)

    abs_d = both_finite["abs_d"]
    lines.append(
        "dFMS (Batch - UI) among finite pairs -- "
        f"max|d|={abs_d.max():.6f}, median|d|={abs_d.median():.6f}, "
        f"mean|d|={abs_d.mean():.6f}, "
        f"exact match (|d|<1e-12)={int((abs_d < 1e-12).sum())}/{len(both_finite)}, "
        f"|d|>=0.01={int((abs_d >= 0.01).sum())}, "
        f"|d|>=0.1={int((abs_d >= 0.1).sum())}"
    )
    rank_ui = both_finite["FMS_UI"].rank(ascending=False)
    rank_batch = both_finite["FMS_Batch"].rank(ascending=False)
    spearman = rank_ui.corr(rank_batch, method="pearson")
    lines.append(f"Rank correlation (finite): {spearman:.4f}")
    return "\n".join(lines)


def inject_staggered_calendar_gaps(
    prices: pd.DataFrame, *, gap_frac: float = 0.08, seed: int = 42
) -> pd.DataFrame:
    """Punch NaN gaps on alternating columns to emulate multi-market calendars."""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    n = len(out)
    for i, col in enumerate(out.columns):
        k = max(1, int(n * gap_frac))
        # Stagger gap blocks by column so markets disagree on holidays
        start = (i * 7) % max(1, n - k)
        idx = out.index[start : start + k]
        # Extra random holes
        extra = rng.choice(out.index, size=min(k, n // 20), replace=False)
        out.loc[idx, col] = np.nan
        out.loc[extra, col] = np.nan
    return out


def split_into_staggered_market_frames(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split columns into two markets with *different native calendars*.

    Market A keeps the full business-day index. Market B drops Mondays and a
    trailing week so per-market ``align_bday_ffill`` + outer concat diverges
    from a single shared-index harmonize (batch).
    """
    cols = list(prices.columns)
    group_a = cols[0::2]
    group_b = cols[1::2]
    frame_a = prices[group_a].copy() if group_a else pd.DataFrame()
    frame_b = prices[group_b].copy() if group_b else pd.DataFrame()
    if not frame_b.empty:
        # Drop Mondays (weekday=0) and last 5 sessions from market B only
        keep = frame_b.index.weekday != 0
        frame_b = frame_b.loc[keep]
        if len(frame_b) > 5:
            frame_b = frame_b.iloc[:-5]
    return frame_a, frame_b


def run_offline(
    prices_path: Path = DEFAULT_PRICES,
    ohlc_path: Path = DEFAULT_OHLC,
    *,
    with_gaps: bool = True,
) -> PathCompareResult:
    """Fixture-based path compare (no network)."""
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    ohlc = None
    if ohlc_path.exists():
        ohlc = pd.read_csv(ohlc_path, header=[0, 1], index_col=0, parse_dates=True)

    local = inject_staggered_calendar_gaps(prices) if with_gaps else prices.copy()
    if with_gaps:
        frame_a, frame_b = split_into_staggered_market_frames(local)
        ui_prices = build_ui_style_from_market_frames(frame_a, frame_b)
        # Batch sees an outer-joined panel (shared download index simulation)
        batch_raw = pd.concat([frame_a, frame_b], axis=1).sort_index()
        batch_raw = batch_raw.loc[:, ~batch_raw.columns.duplicated()]
        ordered = [c for c in local.columns if c in batch_raw.columns]
        batch_prices = build_batch_style_prices_krw(batch_raw[ordered])
    else:
        ui_prices = build_ui_style_from_krw_panel(local)
        batch_prices = build_batch_style_prices_krw(local)
    return compare_fms_paths(ui_prices, batch_prices, ohlc)


def build_ui_style_from_market_frames(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> pd.DataFrame:
    """UI path: align each market on its own calendar, concat, coverage=0.5."""
    frames: list[pd.DataFrame] = []
    if frame_a is not None and not frame_a.empty:
        frames.append(align_bday_ffill(frame_a))
    if frame_b is not None and not frame_b.empty:
        frames.append(align_bday_ffill(frame_b))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1).sort_index()
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return harmonize_calendar(merged, coverage=0.5)


def build_ui_style_from_krw_panel(prices_krw: pd.DataFrame) -> pd.DataFrame:
    """UI-like calendar when the panel is already in KRW (aligned offline).

    Splits columns into odd/even pseudo-markets so per-group ``align_bday_ffill``
    then ``coverage=0.5`` concat can be compared to batch ``coverage=0.9``.
    """
    if prices_krw.empty:
        return prices_krw.copy()
    cols = list(prices_krw.columns)
    frame_a = prices_krw[cols[0::2]] if cols[0::2] else pd.DataFrame()
    frame_b = prices_krw[cols[1::2]] if cols[1::2] else pd.DataFrame()
    return build_ui_style_from_market_frames(frame_a, frame_b)


def _load_watchlist_symbols(mode: str) -> list[str]:
    from watchlist_utils import load_watchlist

    return list(load_watchlist([], mode=mode))


def run_live(
    symbols: Iterable[str],
    *,
    period: str = "1y",
    interval: str = "1d",
    mirror_io: bool = False,
) -> PathCompareResult:
    """Near-simultaneous FMS compare for batch vs UI panel construction.

    Parameters
    ----------
    mirror_io:
        If False (default), download Adj Close **once** and only diverge on
        calendar/FX builders. If True, mirror production I/O: UI = per-market
        downloads; batch = one joint download (still back-to-back in one process).
    """
    symbols = list(dict.fromkeys(str(s) for s in symbols))
    if not symbols:
        raise ValueError("No symbols to compare")

    print("[compare] Downloading FX (shared)...")
    usdkrw, _, jpykrw, hkdkrw = download_fx(period, interval)

    if mirror_io:
        print(
            f"[compare] Mirror-I/O: UI per-market download + batch joint "
            f"download ({len(symbols)} symbols, {period})..."
        )
        usd_symbols = [s for s in symbols if classify(s) == "USA"]
        kor_symbols = [s for s in symbols if classify(s) == "KOR"]
        jpy_symbols = [s for s in symbols if classify(s) == "JPN"]
        hkg_symbols = [s for s in symbols if classify(s) == "HKG"]

        usd_df, _ = download_prices(usd_symbols, period, interval)
        kor_df, _ = download_prices(kor_symbols, period, interval)
        jpy_df, _ = download_prices(jpy_symbols, period, interval)
        hkg_df, _ = download_prices(hkg_symbols, period, interval)

        # Rebuild a local multi-market frame for the UI builder
        parts = [p for p in (usd_df, kor_df, jpy_df, hkg_df) if p is not None and not p.empty]
        if not parts:
            raise RuntimeError("UI per-market download returned empty")
        ui_local = pd.concat(parts, axis=1).sort_index()
        ui_local = ui_local.loc[:, ~ui_local.columns.duplicated()]
        ui_prices = build_ui_style_prices_krw(
            ui_local, usdkrw=usdkrw, jpykrw=jpykrw, hkdkrw=hkdkrw
        )

        batch_local, miss_px = download_prices(symbols, period, interval)
        if miss_px:
            print(f"[compare] Batch joint missing: {len(miss_px)} (e.g. {miss_px[:5]})")
        if batch_local.empty:
            raise RuntimeError("Batch joint download returned empty")
        batch_raw = apply_fx_to_local_prices(
            batch_local, usdkrw=usdkrw, jpykrw=jpykrw, hkdkrw=hkdkrw
        )
        batch_prices = build_batch_style_prices_krw(batch_raw)
        ohlc_symbols = sorted(set(ui_prices.columns) | set(batch_prices.columns))
    else:
        print(f"[compare] Same-download: Adj Close once for {len(symbols)} ({period})...")
        local_prices, miss_px = download_prices(symbols, period, interval)
        if miss_px:
            print(f"[compare] Missing prices: {len(miss_px)} (e.g. {miss_px[:5]})")
        if local_prices.empty:
            raise RuntimeError("Price download returned empty panel")

        print("[compare] Building UI-style panel (align per market, coverage=0.5)...")
        ui_prices = build_ui_style_prices_krw(
            local_prices, usdkrw=usdkrw, jpykrw=jpykrw, hkdkrw=hkdkrw
        )
        print("[compare] Building batch-style panel (shared index, coverage=0.9)...")
        batch_raw = apply_fx_to_local_prices(
            local_prices, usdkrw=usdkrw, jpykrw=jpykrw, hkdkrw=hkdkrw
        )
        batch_prices = build_batch_style_prices_krw(batch_raw)
        ohlc_symbols = list(local_prices.columns)

    print("[compare] Downloading OHLC (tradeability, shared)...")
    ohlc, miss_ohlc = download_ohlc_prices(ohlc_symbols, period, interval)
    if miss_ohlc:
        print(f"[compare] Missing OHLC: {len(miss_ohlc)} (e.g. {miss_ohlc[:5]})")
    if ohlc.empty:
        ohlc = None

    print("[compare] Scoring both paths with identical OHLC...")
    return compare_fms_paths(ui_prices, batch_prices, ohlc)


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    parser = argparse.ArgumentParser(
        description="Quantify FMS d between batch vs UI calendar paths "
        "(near-simultaneous)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--offline",
        action="store_true",
        help="Synthetic fixture path compare (no network)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Download watchlist/symbols and compare paths",
    )
    parser.add_argument(
        "--mode",
        choices=["FREE", "IRP"],
        default="FREE",
        help="Watchlist mode for --live (default FREE)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit tickers for --live (overrides watchlist)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap symbol count after watchlist load (deterministic head)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Rows to print (largest |dFMS| first)",
    )
    parser.add_argument(
        "--no-gaps",
        action="store_true",
        help="Offline only: do not inject calendar gaps",
    )
    parser.add_argument(
        "--mirror-io",
        action="store_true",
        help="LIVE: UI per-market downloads vs batch joint download",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for full comparison table",
    )
    args = parser.parse_args(argv)

    if args.offline:
        print("[compare] Offline fixture path compare...")
        result = run_offline(with_gaps=not args.no_gaps)
    else:
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = _load_watchlist_symbols(args.mode)
            print(f"[compare] Loaded {len(symbols)} symbols from watchlist ({args.mode})")
        if args.limit is not None:
            symbols = symbols[: max(0, args.limit)]
            print(f"[compare] Limited to {len(symbols)} symbols")
        result = run_live(symbols, mirror_io=args.mirror_io)

    print()
    print(summarize_comparison(result))
    print()
    if not result.comparison.empty:
        show = result.comparison.head(args.top)
        with pd.option_context("display.width", 120, "display.max_columns", 12):
            print(show.to_string(float_format=lambda x: f"{x: .6f}"))
    if args.out is not None and not result.comparison.empty:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.comparison.to_csv(args.out)
        print(f"\n[compare] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
