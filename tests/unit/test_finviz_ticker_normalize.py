"""
Unit tests for Finviz ticker first-character dedupe (Validation Harness).

Purpose
-------
Finviz Overview (finvizfinance) may return tickers with the first character
duplicated (``AAPL``→``AAAPL``, ``OKTA``→``OOKTA``). Production normalizes via
``universe_utils.normalize_finviz_tickers`` before writing ``screened_universe.csv``.

Covered behaviors
-----------------
- Strip duplicated leading char when Apple appears as AAAPL (corruption signal)
- No-op when a healthy mix including real ``AAPL`` / ``AA`` is present

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_finviz_ticker_normalize.py -q

No live Finviz/network calls.
"""

from __future__ import annotations

from universe_utils import normalize_finviz_tickers


def test_normalize_finviz_strips_duplicated_first_char_when_apple_corrupted() -> None:
    """AAAPL without AAPL indicates the known Finviz parse bug (first char doubled)."""
    raw = ["AA", "AAA", "AAAPL", "AAMAT", "TTSLA", "OOKTA", "MMSFT"]
    assert normalize_finviz_tickers(raw) == ["A", "AA", "AAPL", "AMAT", "TSLA", "OKTA", "MSFT"]


def test_normalize_finviz_noop_when_tickers_look_healthy() -> None:
    """Healthy mix including AAPL must not be mutated (keep real AA)."""
    raw = ["A", "AA", "AAPL", "MSFT", "TSLA", "OKTA"]
    assert normalize_finviz_tickers(raw) == raw
