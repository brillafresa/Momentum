# -*- coding: utf-8 -*-
"""
Market data port + adapters (HARNESS_RULES.md §2.3).

``MarketDataPort`` is the injection boundary between orchestration
(``calculate_fms_for_batch``) and market data I/O:

- ``YFinanceAdapter``: production adapter delegating to the retry-hardened
  download helpers currently living in ``analysis_utils`` (transitional home;
  the download bodies migrate here later, file by file).
- ``FixtureAdapter``: offline adapter serving pre-loaded price / OHLC / FX
  panels for tests and harness runs. No network I/O.

Contract tests: ``tests/unit/test_market_data_port.py``.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import pandas as pd

from analysis_utils import (
    YF_CHUNK_SIZE_DEFAULT,
    YF_CHUNK_SLEEP_DEFAULT,
    YF_MAX_RETRIES_DEFAULT,
    YF_RATE_LIMIT_INITIAL_SLEEP,
    download_fx,
    download_ohlc_prices,
    download_prices,
)


class MarketDataPort(Protocol):
    """Injectable market data source for batch scoring."""

    def get_prices(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        """Return (Adj Close panel, missing tickers)."""
        ...

    def get_ohlc(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        """Return (MultiIndex (symbol, field) High/Low/Close panel, missing tickers)."""
        ...

    def get_fx(
        self, period_: str, interval: str
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Return (USDKRW, USDJPY, JPYKRW, HKDKRW) series (may be empty)."""
        ...


class YFinanceAdapter:
    """Production adapter: rate-limit-aware yfinance downloads."""

    def __init__(
        self,
        chunk: int = YF_CHUNK_SIZE_DEFAULT,
        chunk_sleep: float = YF_CHUNK_SLEEP_DEFAULT,
        max_retries: int = YF_MAX_RETRIES_DEFAULT,
        initial_sleep: float = YF_RATE_LIMIT_INITIAL_SLEEP,
        threads: bool = False,
    ) -> None:
        self.chunk = chunk
        self.chunk_sleep = chunk_sleep
        self.max_retries = max_retries
        self.initial_sleep = initial_sleep
        self.threads = threads

    def _dl_kwargs(self, period_: str, interval: str) -> dict:
        return dict(
            period_=period_,
            interval=interval,
            chunk=self.chunk,
            chunk_sleep=self.chunk_sleep,
            max_retries=self.max_retries,
            initial_sleep=self.initial_sleep,
            threads=self.threads,
        )

    def get_prices(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        return download_prices(tickers, **self._dl_kwargs(period_, interval))

    def get_ohlc(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        return download_ohlc_prices(tickers, **self._dl_kwargs(period_, interval))

    def get_fx(
        self, period_: str, interval: str
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        return download_fx(period_, interval, initial_sleep=self.initial_sleep)


class FixtureAdapter:
    """Offline adapter: serve pre-loaded panels (tests / harness only)."""

    def __init__(
        self,
        prices: pd.DataFrame,
        ohlc: Optional[pd.DataFrame] = None,
        usdkrw: Optional[pd.Series] = None,
        usdjpy: Optional[pd.Series] = None,
        jpykrw: Optional[pd.Series] = None,
        hkdkrw: Optional[pd.Series] = None,
    ) -> None:
        self._prices = prices
        self._ohlc = ohlc
        self._usdkrw = usdkrw if usdkrw is not None else pd.Series(dtype=float, name='USDKRW')
        self._usdjpy = usdjpy if usdjpy is not None else pd.Series(dtype=float, name='USDJPY')
        self._jpykrw = jpykrw if jpykrw is not None else pd.Series(dtype=float, name='JPYKRW')
        self._hkdkrw = hkdkrw if hkdkrw is not None else pd.Series(dtype=float, name='HKDKRW')

    def get_prices(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        cols = [t for t in tickers if t in self._prices.columns]
        missing = [t for t in tickers if t not in self._prices.columns]
        return self._prices[cols].copy(), missing

    def get_ohlc(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        if self._ohlc is None or self._ohlc.empty:
            return pd.DataFrame(), list(tickers)
        available = set(self._ohlc.columns.get_level_values(0))
        cols = [t for t in tickers if t in available]
        missing = [t for t in tickers if t not in available]
        if not cols:
            return pd.DataFrame(), missing
        return self._ohlc.loc[:, cols].copy(), missing

    def get_fx(
        self, period_: str, interval: str
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        return self._usdkrw, self._usdjpy, self._jpykrw, self._hkdkrw
