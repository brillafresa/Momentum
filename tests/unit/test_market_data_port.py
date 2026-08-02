"""
Unit tests for the MarketDataPort contract (Validation Harness).

Purpose
-------
Lock the port/adapter boundary so batch scoring can run fully offline:

- ``FixtureAdapter`` serves pre-loaded price / OHLC / FX panels and reports
  missing tickers without any network I/O.
- ``calculate_fms_for_batch`` accepts an injected ``market_data`` port and,
  given the same fixture panels, reproduces the direct
  ``momentum_now_and_delta`` scores (download / score separation).
- ``YFinanceAdapter`` delegates to the retry-hardened download helpers in
  ``analysis_utils`` with its configured chunk/retry settings (mocked).

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_market_data_port.py -q

Fixtures: ``tests/fixtures/synthetic_*.csv``. No live market API.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from adapters.market_data import FixtureAdapter, YFinanceAdapter
from analysis_utils import calculate_fms_for_batch, momentum_now_and_delta


@pytest.fixture()
def fixture_adapter(synthetic_prices_krw: pd.DataFrame, synthetic_ohlc: pd.DataFrame) -> FixtureAdapter:
    return FixtureAdapter(prices=synthetic_prices_krw, ohlc=synthetic_ohlc)


def test_fixture_adapter_returns_requested_columns_and_missing(
    fixture_adapter: FixtureAdapter,
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Port must slice known tickers and report unknown ones as missing."""
    known = list(synthetic_prices_krw.columns[:2])
    prices, missing = fixture_adapter.get_prices(known + ["NOPE"], "1y", "1d")
    assert list(prices.columns) == known
    assert missing == ["NOPE"]

    ohlc, ohlc_missing = fixture_adapter.get_ohlc(known + ["NOPE"], "1y", "1d")
    assert sorted(set(ohlc.columns.get_level_values(0))) == sorted(known)
    assert ohlc_missing == ["NOPE"]


def test_fixture_adapter_fx_defaults_to_empty(fixture_adapter: FixtureAdapter) -> None:
    """Without FX fixtures the port returns empty series (panel already in KRW)."""
    usdkrw, usdjpy, jpykrw, hkdkrw = fixture_adapter.get_fx("1y", "1d")
    assert usdkrw.empty and usdjpy.empty and jpykrw.empty and hkdkrw.empty


def test_batch_with_fixture_adapter_matches_direct_scoring_offline(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
    fixture_adapter: FixtureAdapter,
) -> None:
    """Injected port batch must reproduce direct momentum_now_and_delta FMS.

    yfinance.download is patched to fail: the batch path must be fully offline
    when a port is injected (download / score separation).
    """
    symbols = list(synthetic_prices_krw.columns)
    with patch("yfinance.download", side_effect=AssertionError("network forbidden")):
        batch = calculate_fms_for_batch(
            symbols,
            reference_prices_krw=synthetic_prices_krw,
            market_data=fixture_adapter,
        )
    direct = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=symbols,
    )
    assert sorted(batch.index) == sorted(direct.index)
    pd.testing.assert_series_equal(
        batch["FMS"].sort_index(), direct["FMS"].sort_index(), check_names=False
    )
    assert list(batch.index) == list(direct.index)  # same FMS-descending order


def test_batch_fms_is_invariant_to_account_watchlist_reference(
    synthetic_prices_krw: pd.DataFrame,
    fixture_adapter: FixtureAdapter,
) -> None:
    """v5.0.0 absolute FMS: reference panel must not change candidate scores."""
    symbols = list(synthetic_prices_krw.columns)
    weak_reference = synthetic_prices_krw[["FLAT", "CRASHY"]]
    strong_reference = synthetic_prices_krw[["TREND_UP", "MILD_UP"]]

    weak_ref_scores = calculate_fms_for_batch(
        symbols,
        reference_prices_krw=weak_reference,
        market_data=fixture_adapter,
    )
    strong_ref_scores = calculate_fms_for_batch(
        symbols,
        reference_prices_krw=strong_reference,
        market_data=fixture_adapter,
    )

    pd.testing.assert_series_equal(
        weak_ref_scores["FMS"].sort_index(),
        strong_ref_scores["FMS"].sort_index(),
        check_names=False,
    )


def test_self_referenced_batch_does_not_normalize_per_outer_chunk(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
    fixture_adapter: FixtureAdapter,
) -> None:
    """Watchlist reassessment uses one full self-reference even with outer=2."""
    symbols = list(synthetic_prices_krw.columns)
    batch = calculate_fms_for_batch(
        symbols,
        reference_prices_krw=None,
        outer_batch_size=2,
        market_data=fixture_adapter,
    )
    direct = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=symbols,
    )
    pd.testing.assert_series_equal(
        batch["FMS"].sort_index(), direct["FMS"].sort_index(), check_names=False
    )


def test_yfinance_adapter_delegates_with_configured_settings() -> None:
    """YFinanceAdapter must call analysis_utils download helpers with its config."""
    adapter = YFinanceAdapter(chunk=7, chunk_sleep=0.9, max_retries=3)
    sentinel = (pd.DataFrame({"AAPL": [1.0]}), [])

    with patch("adapters.market_data.download_prices", return_value=sentinel) as mock_px:
        out = adapter.get_prices(["AAPL"], "1y", "1d")
    assert out is sentinel
    kwargs = mock_px.call_args.kwargs
    assert kwargs["chunk"] == 7
    assert kwargs["chunk_sleep"] == 0.9
    assert kwargs["max_retries"] == 3

    with patch("adapters.market_data.download_ohlc_prices", return_value=sentinel) as mock_ohlc:
        adapter.get_ohlc(["AAPL"], "1y", "1d")
    assert mock_ohlc.call_args.kwargs["chunk"] == 7

    fx_sentinel = (
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
    )
    with patch("adapters.market_data.download_fx", return_value=fx_sentinel) as mock_fx:
        adapter.get_fx("1y", "1d")
    assert mock_fx.call_args.args[0] == "1y"
