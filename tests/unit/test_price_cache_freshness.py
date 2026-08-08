# -*- coding: utf-8 -*-
"""Unit tests for last-bar freshness and CachingMarketDataAdapter.

Purpose
-------
Validate disk cache HIT/MISS rules (same calendar date = HIT; newer probe =
refresh; probe failure keeps cache) and that the caching adapter avoids a
second full download after write-through (FixtureAdapter call counts).

Run
---
    python -m pytest tests/unit/test_price_cache_freshness.py -q

Note: uses a temp ``DiskPriceCache`` root — never the production ``cache/`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adapters.market_data import FixtureAdapter
from adapters.price_cache import (
    CachingMarketDataAdapter,
    DiskPriceCache,
    last_bar_date,
    needs_refresh,
)


def test_needs_refresh_empty_cache():
    assert needs_refresh(None, pd.Timestamp("2024-06-10")) is True


def test_needs_refresh_same_day_is_hit():
    d = pd.Timestamp("2024-06-10")
    assert needs_refresh(d, d) is False
    assert needs_refresh(d, pd.Timestamp("2024-06-10 15:30")) is False


def test_needs_refresh_newer_probe():
    assert needs_refresh(pd.Timestamp("2024-06-09"), pd.Timestamp("2024-06-10")) is True


def test_needs_refresh_probe_none_keeps_cache():
    assert needs_refresh(pd.Timestamp("2024-06-09"), None) is False


def test_last_bar_date_series_and_empty():
    idx = pd.bdate_range("2024-01-01", periods=3)
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    assert last_bar_date(s) == idx[-1]
    assert last_bar_date(pd.Series(dtype=float)) is None
    assert last_bar_date(None) is None


class _CountingFixture(FixtureAdapter):
    """FixtureAdapter that counts get_prices / get_ohlc calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.price_calls = 0
        self.ohlc_calls = 0

    def get_prices(self, tickers, period_, interval):
        self.price_calls += 1
        return super().get_prices(tickers, period_, interval)

    def get_ohlc(self, tickers, period_, interval):
        self.ohlc_calls += 1
        return super().get_ohlc(tickers, period_, interval)


def _sample_prices():
    idx = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame({"AAA": range(10, 20), "BBB": range(20, 30)}, index=idx)


def test_caching_adapter_second_call_hits_disk(tmp_path: Path):
    prices = _sample_prices()
    inner = _CountingFixture(prices=prices)
    cache = DiskPriceCache(root=tmp_path / "md")
    adapter = CachingMarketDataAdapter(inner, cache=cache, probe_period="5d")

    # First call: probe + full miss → inner get_prices called twice (probe + full)
    out1, miss1 = adapter.get_prices(["AAA", "BBB"], "1y", "1d")
    assert miss1 == []
    assert list(out1.columns) == ["AAA", "BBB"]
    calls_after_first = inner.price_calls
    assert calls_after_first >= 2

    # Second call: probe still hits inner, but full download skipped (disk HIT)
    out2, miss2 = adapter.get_prices(["AAA", "BBB"], "1y", "1d")
    assert miss2 == []
    assert list(out2.columns) == ["AAA", "BBB"]
    # Only probe call(s) added — no second full miss download for both
    assert inner.price_calls == calls_after_first + 1
    assert adapter.stats["hits"] >= 2


def test_disk_cache_roundtrip(tmp_path: Path):
    cache = DiskPriceCache(root=tmp_path / "md")
    idx = pd.bdate_range("2024-01-01", periods=5)
    s = pd.Series(range(5), index=idx, name="AAA")
    cache.save_symbol_series("AAA", s, kind="adj_close", period="1y")
    loaded = cache.load_symbol_series("AAA", "adj_close")
    assert loaded is not None
    assert last_bar_date(loaded) == idx[-1]
    assert cache.cached_last_bar("AAA") == idx[-1]
