# -*- coding: utf-8 -*-
"""Disk-backed market data cache with last-bar freshness probes.

Shared by batch (write-through) and UI (read + probe). No FMS formulas here.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_CACHE_ROOT = Path("cache") / "market_data"


def last_bar_date(obj: Optional[pd.Series | pd.DataFrame]) -> Optional[pd.Timestamp]:
    """Return the last valid timestamp on a Series or DataFrame, or None."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        if obj.empty:
            return None
        lv = obj.last_valid_index()
        return pd.Timestamp(lv) if lv is not None else None
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return None
        last = None
        for col in obj.columns:
            lv = obj[col].last_valid_index()
            if lv is None:
                continue
            ts = pd.Timestamp(lv)
            if last is None or ts > last:
                last = ts
        return last
    return None


def needs_refresh(
    cached_last: Optional[pd.Timestamp],
    probed_last: Optional[pd.Timestamp],
) -> bool:
    """True when full-period download should replace the cache entry.

    - No cache → refresh
    - Probe failed (None) but cache exists → keep cache (stale HIT)
    - Probe date strictly newer than cache → refresh
    - Same calendar date → HIT (intraday updates ignored for daily bars)
    """
    if cached_last is None:
        return True
    if probed_last is None:
        return False
    c = pd.Timestamp(cached_last).normalize()
    p = pd.Timestamp(probed_last).normalize()
    return p > c


def _safe_symbol_filename(symbol: str) -> str:
    """Filesystem-safe token for a ticker symbol."""
    return re.sub(r"[^\w.\-]+", "_", str(symbol))


class DiskPriceCache:
    """Per-symbol parquet store under ``cache/market_data/``."""

    def __init__(self, root: Optional[os.PathLike | str] = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"symbols": {}}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"symbols": {}}
            data.setdefault("symbols", {})
            return data
        except (OSError, json.JSONDecodeError):
            return {"symbols": {}}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2, default=str)

    def _paths(self, symbol: str, kind: str) -> Tuple[Path, Path]:
        base = _safe_symbol_filename(symbol)
        return self.root / f"{base}__{kind}.parquet", self.root / f"{base}__{kind}.meta.json"

    def load_symbol_series(self, symbol: str, kind: str = "adj_close") -> Optional[pd.DataFrame]:
        """Load cached panel for one symbol. Adj close → 1-col DF; OHLC → MultiIndex cols."""
        path, _ = self._paths(symbol, kind)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError):
            return None
        if df.empty:
            return None
        return df

    def save_symbol_series(
        self,
        symbol: str,
        data: pd.DataFrame | pd.Series,
        kind: str = "adj_close",
        period: str = "1y",
    ) -> None:
        """Persist one symbol's data and update manifest last_bar."""
        if data is None or (hasattr(data, "empty") and data.empty):
            return
        if isinstance(data, pd.Series):
            df = data.to_frame(name=str(symbol))
        else:
            df = data.copy()
        path, meta_path = self._paths(symbol, kind)
        df.to_parquet(path)
        lb = last_bar_date(df)
        meta = {
            "symbol": str(symbol),
            "kind": kind,
            "period": period,
            "last_bar": lb.isoformat() if lb is not None else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(df)),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._manifest.setdefault("symbols", {})
        self._manifest["symbols"].setdefault(str(symbol), {})[kind] = meta
        self._save_manifest()

    def cached_last_bar(self, symbol: str, kind: str = "adj_close") -> Optional[pd.Timestamp]:
        """Return last_bar from manifest or by loading the parquet."""
        entry = self._manifest.get("symbols", {}).get(str(symbol), {}).get(kind)
        if entry and entry.get("last_bar"):
            try:
                return pd.Timestamp(entry["last_bar"])
            except (TypeError, ValueError):
                pass
        df = self.load_symbol_series(symbol, kind)
        return last_bar_date(df)

    def save_price_panel(self, prices: pd.DataFrame, period: str = "1y") -> None:
        """Write each column of an Adj Close panel as its own cache entry."""
        if prices is None or prices.empty:
            return
        for col in prices.columns:
            self.save_symbol_series(str(col), prices[col], kind="adj_close", period=period)

    def save_ohlc_panel(self, ohlc: pd.DataFrame, period: str = "1y") -> None:
        """Write each symbol block of a MultiIndex OHLC panel."""
        if ohlc is None or ohlc.empty:
            return
        if not isinstance(ohlc.columns, pd.MultiIndex):
            return
        symbols = list(ohlc.columns.get_level_values(0).unique())
        for sym in symbols:
            try:
                block = ohlc[sym]
            except KeyError:
                continue
            # Store with MultiIndex columns (sym, field) for round-trip
            if isinstance(block, pd.Series):
                continue
            wide = block.copy()
            wide.columns = pd.MultiIndex.from_product([[str(sym)], wide.columns])
            self.save_symbol_series(str(sym), wide, kind="ohlc", period=period)


class CachingMarketDataAdapter:
    """MarketDataPort wrapper: last-bar probe → disk HIT or full download + save."""

    def __init__(
        self,
        inner,
        cache: Optional[DiskPriceCache] = None,
        probe_period: str = "5d",
        write_through: bool = True,
    ) -> None:
        self.inner = inner
        self.cache = cache if cache is not None else DiskPriceCache()
        self.probe_period = probe_period
        self.write_through = write_through
        self.stats = {"hits": 0, "misses": 0, "stale_hits": 0, "probes": 0}

    def _probe_last_bars(self, tickers: Sequence[str], interval: str) -> dict:
        """Download short history and map symbol → last bar date."""
        tickers = [str(t) for t in tickers]
        if not tickers:
            return {}
        self.stats["probes"] += 1
        try:
            panel, _ = self.inner.get_prices(list(tickers), self.probe_period, interval)
        except Exception:
            return {}
        out = {}
        for t in tickers:
            if panel is not None and not panel.empty and t in panel.columns:
                out[t] = last_bar_date(panel[t])
            else:
                out[t] = None
        return out

    def get_prices(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        tickers = [str(t) for t in tickers]
        if not tickers:
            return pd.DataFrame(), []

        probed = self._probe_last_bars(tickers, interval)
        hit_cols = []
        miss_syms = []
        frames = []

        for t in tickers:
            cached = self.cache.load_symbol_series(t, "adj_close")
            cached_last = last_bar_date(cached) if cached is not None else self.cache.cached_last_bar(t, "adj_close")
            probed_last = probed.get(t)
            refresh = needs_refresh(cached_last, probed_last)
            if not refresh and cached is not None and not cached.empty:
                if probed_last is None and cached_last is not None:
                    self.stats["stale_hits"] += 1
                else:
                    self.stats["hits"] += 1
                col = cached.iloc[:, 0] if cached.shape[1] >= 1 else cached.squeeze()
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                col = col.copy()
                col.name = t
                frames.append(col)
                hit_cols.append(t)
            else:
                miss_syms.append(t)
                self.stats["misses"] += 1

        missing: List[str] = []
        if miss_syms:
            fresh, miss = self.inner.get_prices(miss_syms, period_, interval)
            missing.extend(miss)
            if fresh is not None and not fresh.empty:
                if self.write_through:
                    self.cache.save_price_panel(fresh, period=period_)
                for c in fresh.columns:
                    frames.append(fresh[c].copy())

        if not frames:
            return pd.DataFrame(), list(dict.fromkeys(missing + miss_syms))

        out = pd.concat(frames, axis=1).sort_index()
        out = out.loc[:, ~out.columns.duplicated()]
        # Preserve request order where possible
        ordered = [t for t in tickers if t in out.columns]
        out = out[ordered]
        still_missing = [t for t in tickers if t not in out.columns]
        missing = list(dict.fromkeys(missing + still_missing))
        return out, missing

    def get_ohlc(self, tickers: List[str], period_: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        tickers = [str(t) for t in tickers]
        if not tickers:
            return pd.DataFrame(), []

        # Probe via Adj Close last bars (same calendar freshness signal)
        probed = self._probe_last_bars(tickers, interval)
        hit_blocks = []
        miss_syms = []

        for t in tickers:
            cached = self.cache.load_symbol_series(t, "ohlc")
            cached_last = last_bar_date(cached) if cached is not None else self.cache.cached_last_bar(t, "ohlc")
            probed_last = probed.get(t)
            refresh = needs_refresh(cached_last, probed_last)
            if not refresh and cached is not None and not cached.empty:
                if probed_last is None and cached_last is not None:
                    self.stats["stale_hits"] += 1
                else:
                    self.stats["hits"] += 1
                hit_blocks.append(cached)
            else:
                miss_syms.append(t)
                self.stats["misses"] += 1

        missing: List[str] = []
        if miss_syms:
            fresh, miss = self.inner.get_ohlc(miss_syms, period_, interval)
            missing.extend(miss)
            if fresh is not None and not fresh.empty:
                if self.write_through:
                    self.cache.save_ohlc_panel(fresh, period=period_)
                hit_blocks.append(fresh)

        if not hit_blocks:
            return pd.DataFrame(), list(dict.fromkeys(missing + miss_syms))

        out = pd.concat(hit_blocks, axis=1).sort_index()
        # Drop duplicate symbol blocks if any
        if isinstance(out.columns, pd.MultiIndex):
            seen = set()
            keep = []
            for col in out.columns:
                sym = col[0]
                if sym in seen and col in keep:
                    continue
                key = (col[0], col[1]) if isinstance(col, tuple) else col
                if key in seen:
                    continue
                seen.add(key)
                keep.append(col)
            out = out.loc[:, keep]
        still_missing = []
        if isinstance(out.columns, pd.MultiIndex):
            available = set(out.columns.get_level_values(0))
            still_missing = [t for t in tickers if t not in available]
        missing = list(dict.fromkeys(missing + still_missing))
        return out, missing

    def get_fx(
        self, period_: str, interval: str
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """FX via inner adapter; write-through raw/derived series to disk when enabled."""
        usdkrw, usdjpy, jpykrw, hkdkrw = self.inner.get_fx(period_, interval)
        if self.write_through:
            for sym, series in (
                ("KRW=X", usdkrw),
                ("JPY=X", usdjpy),
                ("JPYKRW", jpykrw),
                ("HKDKRW", hkdkrw),
            ):
                if series is not None and not getattr(series, "empty", True):
                    self.cache.save_symbol_series(sym, series, kind="adj_close", period=period_)
        return usdkrw, usdjpy, jpykrw, hkdkrw


def make_caching_yfinance_adapter(**kwargs) -> CachingMarketDataAdapter:
    """Production helper: YFinanceAdapter wrapped with disk last-bar cache."""
    from adapters.market_data import YFinanceAdapter

    return CachingMarketDataAdapter(YFinanceAdapter(**kwargs))


def clear_disk_market_cache(root: Optional[os.PathLike | str] = None) -> None:
    """Delete parquet/meta/manifest under the market-data cache root."""
    root_path = Path(root) if root is not None else DEFAULT_CACHE_ROOT
    if not root_path.exists():
        return
    for path in root_path.rglob("*"):
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
