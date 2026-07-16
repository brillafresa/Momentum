"""
Unit tests for Yahoo rate-limit detection / retry (Validation Harness).

Purpose
-------
yfinance multi-download swallows ``YFRateLimitError`` into ``shared._ERRORS``
instead of raising. These tests lock that detection + retry path so batch scans
do not silently drop chunks under 429.

Covered behaviors
-----------------
- Message heuristics for Too Many Requests / 429 / YFRateLimitError
- Retry when ``_pop_yf_shared_rate_limited`` reports hits (no live network)

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_yf_rate_limit_retry.py -q
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from analysis_utils import _is_rate_limit_message, _yf_download_with_retry


def test_is_rate_limit_message_detects_known_forms() -> None:
    """Rate-limit heuristics must catch yfinance / HTTP variants."""
    assert _is_rate_limit_message("Too Many Requests. Rate limited. Try after a while.")
    assert _is_rate_limit_message("YFRateLimitError()")
    assert _is_rate_limit_message("HTTP 429")
    assert not _is_rate_limit_message("YFPricesMissingError('possibly delisted')")


def test_yf_download_retries_when_shared_errors_report_rate_limit() -> None:
    """
    yfinance swallows YFRateLimitError into shared._ERRORS; wrapper must retry.

    First call: empty frame + rate-limit error in shared state.
    Second call: non-empty frame without rate-limit errors.
    """
    empty = pd.DataFrame()
    ok = pd.DataFrame({"Close": [1.0, 2.0]})

    with patch("analysis_utils.yf.download", side_effect=[empty, ok]) as mock_dl, \
         patch("analysis_utils._clear_yf_shared_state"), \
         patch("analysis_utils._pop_yf_shared_rate_limited", side_effect=[["AAPL"], []]), \
         patch("analysis_utils.time.sleep") as mock_sleep:
        out = _yf_download_with_retry(
            ["AAPL"], period_="1y", interval="1d",
            max_retries=5, initial_sleep=2.0, threads=False,
        )

    assert out is ok
    assert mock_dl.call_count == 2
    assert mock_sleep.call_count >= 1
