# -*- coding: utf-8 -*-
"""LIVE smoke: multi-market FMS batch via CachingMarketDataAdapter.

Purpose
-------
Reproduce / guard the v5.0.3 side effect where aggressive last-bar probes
amplified Yahoo rate limits and early (USA) outer chunks failed, leaving
신규 종목 탐색 with only KOR/HKG rows.

This runner scores a small FREE-like mix (USA + KOR + HKG) through the same
caching adapter path as ``run_scan_batch`` and asserts all three markets
appear in the scored frame.

Run (LIVE network; not for CI)
------------------------------
    python -m harness.smoke_multi_market_batch
    python -m harness.smoke_multi_market_batch --tmpdir cache/_smoke_md

Exit 0 only when USA, KOR, and HKG each contribute ≥1 scored row.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

from adapters.price_cache import CachingMarketDataAdapter, DiskPriceCache
from analysis_utils import calculate_fms_for_batch, classify


DEFAULT_SYMBOLS = [
    # USA
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "KO",
    # KOR
    "005930.KS",
    "000660.KS",
    "035420.KS",
    # HKG
    "0700.HK",
    "0005.HK",
    "9988.HK",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tmpdir",
        default="",
        help="Disk cache root (default: fresh temp dir, not production cache/)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=DEFAULT_SYMBOLS,
        help="Symbols to score (default: small USA+KOR+HKG mix)",
    )
    args = parser.parse_args(argv)

    if args.tmpdir:
        cache_root = Path(args.tmpdir)
        cache_root.mkdir(parents=True, exist_ok=True)
    else:
        cache_root = Path(tempfile.mkdtemp(prefix="momentum_smoke_md_"))

    print(f"[smoke] cache root: {cache_root}")
    print(f"[smoke] symbols ({len(args.symbols)}): {args.symbols}")

    from adapters.market_data import YFinanceAdapter

    inner = YFinanceAdapter(threads=False)
    adapter = CachingMarketDataAdapter(inner, cache=DiskPriceCache(root=cache_root))
    results = calculate_fms_for_batch(
        list(args.symbols),
        period_="1y",
        market_data=adapter,
        outer_batch_size=max(len(args.symbols), 1),
    )
    if results.empty:
        print("[smoke] FAIL: empty results")
        print(f"[smoke] adapter stats: {adapter.stats}")
        return 1

    markets = Counter(classify(str(i)) for i in results.index)
    print(f"[smoke] scored {len(results)} rows by market: {dict(markets)}")
    print(f"[smoke] adapter stats: {adapter.stats}")
    for m in ("USA", "KOR", "HKG"):
        n = markets.get(m, 0)
        print(f"[smoke]   {m}: {n}")
        if n < 1:
            print(f"[smoke] FAIL: missing market {m}")
            return 1

    # Second pass should probe warm entries only (no cold full re-download for all)
    adapter2 = CachingMarketDataAdapter(inner, cache=DiskPriceCache(root=cache_root))
    results2 = calculate_fms_for_batch(
        list(args.symbols),
        period_="1y",
        market_data=adapter2,
        outer_batch_size=max(len(args.symbols), 1),
    )
    print(f"[smoke] second pass scored {len(results2)}; stats={adapter2.stats}")
    if results2.empty:
        print("[smoke] FAIL: second pass empty")
        return 1
    if adapter2.stats.get("cold_misses", 0) > 0 and adapter2.stats.get("hits", 0) == 0:
        print("[smoke] WARN: second pass did not HIT disk; check probe/write-through")

    print("[smoke] OK: USA + KOR + HKG all present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
