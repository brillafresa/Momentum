# -*- coding: utf-8 -*-
"""UI watchlist / detail-view session bundles (no network I/O).

Fingerprint helpers and ``DetailViewAtom`` keep detail-view labels bound to the
exact price series used for charts so ticker/index drift cannot silently
mismatch chart data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def _ts_key(ts: Any) -> Optional[str]:
    """Normalize a timestamp-like value to a stable string key."""
    if ts is None:
        return None
    try:
        if pd.isna(ts):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(ts).isoformat()
    except (TypeError, ValueError):
        return str(ts)


def fingerprint_price_panel(prices_krw: pd.DataFrame) -> Tuple:
    """Return a hashable fingerprint of a KRW price panel (no full DataFrame hash).

    Uses column order, per-column last valid index, and non-null counts so a
    newer bar or column set changes the fingerprint without hashing values.
    """
    if prices_krw is None or prices_krw.empty:
        return ("empty",)
    cols = tuple(str(c) for c in prices_krw.columns)
    parts: List[Tuple[str, Optional[str], int]] = []
    for col in prices_krw.columns:
        series = prices_krw[col]
        last = series.last_valid_index()
        n_valid = int(series.notna().sum())
        parts.append((str(col), _ts_key(last), n_valid))
    return ("prices", cols, tuple(parts))


def fingerprint_ohlc_panel(ohlc_data: Optional[pd.DataFrame]) -> Tuple:
    """Fingerprint an OHLC MultiIndex panel (symbol, field) or empty marker."""
    if ohlc_data is None or ohlc_data.empty:
        return ("ohlc_empty",)
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        symbols = tuple(str(s) for s in ohlc_data.columns.get_level_values(0).unique())
    else:
        symbols = tuple(str(c) for c in ohlc_data.columns)
    parts: List[Tuple[str, Optional[str], int]] = []
    for sym in symbols:
        try:
            block = ohlc_data[sym] if isinstance(ohlc_data.columns, pd.MultiIndex) else ohlc_data[[sym]]
        except (KeyError, TypeError):
            continue
        if isinstance(block, pd.DataFrame):
            last = None
            n_valid = 0
            for col in block.columns:
                s = block[col]
                lv = s.last_valid_index()
                if last is None or (lv is not None and (last is None or lv > last)):
                    last = lv
                n_valid = max(n_valid, int(s.notna().sum()))
        else:
            last = block.last_valid_index()
            n_valid = int(block.notna().sum())
        parts.append((str(sym), _ts_key(last), n_valid))
    return ("ohlc", symbols, tuple(parts))


def watchlist_bundle_key(
    account_mode: str,
    watchlist: Sequence[str],
    min_data_period: str,
    prices_krw: pd.DataFrame,
    ohlc_data: Optional[pd.DataFrame] = None,
) -> Tuple:
    """Session memo key for the full watchlist price/OHLC/FMS bundle."""
    return (
        str(account_mode),
        tuple(str(s) for s in watchlist),
        str(min_data_period),
        fingerprint_price_panel(prices_krw),
        fingerprint_ohlc_panel(ohlc_data),
    )


@dataclass(frozen=True)
class DetailViewAtom:
    """Atomic detail-view payload: symbol identity bound to its price series."""

    symbol: str
    display_name: str
    series_krw: pd.Series
    period_key: str
    fingerprint: Tuple
    fms_row: Optional[pd.Series] = None

    def is_consistent(self) -> bool:
        """True iff series name and optional FMS row match ``symbol``."""
        if self.series_krw is None or self.series_krw.name != self.symbol:
            return False
        if self.fms_row is not None and self.fms_row.name != self.symbol:
            return False
        return True


def build_detail_view_atom(
    symbol: str,
    prices_krw: pd.DataFrame,
    name_map: Optional[Mapping[str, str]] = None,
    mom: Optional[pd.DataFrame] = None,
    period_key: str = "6M",
) -> Optional[DetailViewAtom]:
    """Build a DetailViewAtom for ``symbol`` or return None if missing.

    Never silently substitutes another column when ``symbol`` is absent.
    """
    if symbol is None or prices_krw is None or prices_krw.empty:
        return None
    sym = str(symbol)
    if sym not in prices_krw.columns:
        return None

    series = prices_krw[sym].copy()
    series.name = sym

    display = sym
    if name_map:
        mapped = name_map.get(sym)
        if mapped:
            display = str(mapped)

    fms_row: Optional[pd.Series] = None
    if mom is not None and not mom.empty and sym in mom.index:
        fms_row = mom.loc[sym].copy()
        fms_row.name = sym

    panel_fp = fingerprint_price_panel(prices_krw[[sym]])
    atom = DetailViewAtom(
        symbol=sym,
        display_name=display,
        series_krw=series,
        period_key=str(period_key),
        fingerprint=panel_fp,
        fms_row=fms_row,
    )
    if not atom.is_consistent():
        return None
    return atom


def reconcile_detail_selection(
    ordered_options: Sequence[str],
    current_symbol: Optional[str],
    default_sym: str,
) -> Tuple[str, int]:
    """Return (valid_symbol, index) for detail select against current options.

    If ``current_symbol`` is missing or not in options, fall back to ``default_sym``
    (or first option). Index is always derived from the resolved symbol.
    """
    options = [str(s) for s in ordered_options]
    if not options:
        return str(default_sym), 0

    default = str(default_sym) if str(default_sym) in options else options[0]
    if current_symbol is not None and str(current_symbol) in options:
        sym = str(current_symbol)
    else:
        sym = default
    return sym, options.index(sym)


SESSION_BUNDLE_KEY = "ui_watchlist_bundle"
SESSION_BUNDLE_FP_KEY = "ui_watchlist_bundle_fp"
DETAIL_ATOM_CACHE_KEY = "ui_detail_atom_cache"
DETAIL_SELECT_KEY = "detail_selectbox"
DETAIL_INDEX_KEY = "detail_symbol_index"


def clear_ui_session_caches(session_state: Dict[str, Any]) -> None:
    """Drop watchlist memo + detail atom caches from a session_state mapping."""
    for key in (
        SESSION_BUNDLE_KEY,
        SESSION_BUNDLE_FP_KEY,
        DETAIL_ATOM_CACHE_KEY,
        DETAIL_SELECT_KEY,
        DETAIL_INDEX_KEY,
    ):
        if key in session_state:
            del session_state[key]
