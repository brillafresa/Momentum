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
    cache_covers_request,
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


def _sample_prices(n: int = 200):
    """Enough bars to satisfy the 1y HIT floor (``_PERIOD_MIN_BARS``)."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"AAA": range(n, 2 * n), "BBB": range(2 * n, 3 * n)}, index=idx)


def test_caching_adapter_cold_path_skips_probe(tmp_path: Path):
    """No disk entry → one full get_prices only (no 5d probe round-trip)."""
    prices = _sample_prices()
    inner = _CountingFixture(prices=prices)
    cache = DiskPriceCache(root=tmp_path / "md")
    adapter = CachingMarketDataAdapter(inner, cache=cache, probe_period="5d")

    out1, miss1 = adapter.get_prices(["AAA", "BBB"], "1y", "1d")
    assert miss1 == []
    assert list(out1.columns) == ["AAA", "BBB"]
    assert inner.price_calls == 1
    assert adapter.stats["probes"] == 0
    assert adapter.stats["cold_misses"] == 2


def test_caching_adapter_second_call_hits_disk(tmp_path: Path):
    prices = _sample_prices()
    inner = _CountingFixture(prices=prices)
    cache = DiskPriceCache(root=tmp_path / "md")
    adapter = CachingMarketDataAdapter(inner, cache=cache, probe_period="5d")

    # Cold: full download only
    out1, miss1 = adapter.get_prices(["AAA", "BBB"], "1y", "1d")
    assert miss1 == []
    assert list(out1.columns) == ["AAA", "BBB"]
    assert inner.price_calls == 1

    # Warm: probe cached symbols only, then HIT (no second full download)
    out2, miss2 = adapter.get_prices(["AAA", "BBB"], "1y", "1d")
    assert miss2 == []
    assert list(out2.columns) == ["AAA", "BBB"]
    assert inner.price_calls == 2  # +1 probe
    assert adapter.stats["probes"] == 1
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
    assert cache.cached_period("AAA") == "1y"


def test_cache_covers_request_period_rank_and_bars():
    idx_1y = pd.bdate_range("2024-01-01", periods=200)
    short = pd.DataFrame({"AAA": range(200)}, index=idx_1y)
    assert cache_covers_request(short, "1y", "1y") is True
    assert cache_covers_request(short, "2y", "1y") is False  # meta period shorter
    assert cache_covers_request(short, "2y", None) is False  # bar floor for 2y

    idx_2y = pd.bdate_range("2023-01-01", periods=400)
    long = pd.DataFrame({"AAA": range(400)}, index=idx_2y)
    assert cache_covers_request(long, "2y", "2y") is True
    assert cache_covers_request(long, "2y", "1y") is False  # meta still wins


def test_caching_adapter_period_miss_refreshes_for_longer_request(tmp_path: Path):
    """Batch-like 1y cache must not HIT when UI asks for 2y (ITGR coverage bug)."""
    idx_1y = pd.bdate_range("2024-06-01", periods=200)
    idx_2y = pd.bdate_range("2023-06-01", periods=400)
    # Inner serves 2y panel when asked; cache starts with 1y-only write.
    prices_2y = pd.DataFrame({"ITGR": range(400)}, index=idx_2y)
    inner = _CountingFixture(prices=prices_2y)
    cache = DiskPriceCache(root=tmp_path / "md")
    cache.save_symbol_series(
        "ITGR",
        pd.Series(range(200), index=idx_1y, name="ITGR"),
        kind="adj_close",
        period="1y",
    )
    adapter = CachingMarketDataAdapter(inner, cache=cache, probe_period="5d")

    out, miss = adapter.get_prices(["ITGR"], "2y", "1d")
    assert miss == []
    assert "ITGR" in out.columns
    assert len(out["ITGR"].dropna()) >= 360
    assert adapter.stats["period_misses"] == 1
    assert adapter.stats["probes"] == 0  # period miss skips probe
    assert inner.price_calls == 1  # full 2y download
    assert cache.cached_period("ITGR") == "2y"
