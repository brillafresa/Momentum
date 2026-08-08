# -*- coding: utf-8 -*-
"""LIVE: USA-first medium batch through caching adapter (ops reproduction).

Purpose
-------
Mimic FREE universe order (Finviz USA first, then KOR/HKG) under the production
``CachingMarketDataAdapter`` path and assert USA rows survive scoring / FMS≥0 filter.

This is the regression harness for the v5.0.4 warm-only probe fix: cold-path
probing used to amplify Yahoo rate limits so early USA outer chunks failed
while later KR/HK succeeded.

Run (LIVE network; not for CI)
------------------------------
    python -m harness.smoke_usa_first_batch
    python -m harness.smoke_usa_first_batch --save   # optional: write scan_results/

Default uses a **temp disk-cache root** (never production ``cache/``).
``--save`` is opt-in; without it ``scan_results/`` is untouched.

Exit 0 only when USA, KOR, and HKG each contribute ≥1 scored row and ≥1 row
with FMS ≥ DEFAULT_FMS_THRESHOLD.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from adapters.market_data import YFinanceAdapter
from adapters.price_cache import CachingMarketDataAdapter, DiskPriceCache
from analysis_utils import calculate_fms_for_batch, classify
from config import DEFAULT_FMS_THRESHOLD
from universe_utils import save_scan_results, MODE_FREE


def _load_usa(n: int = 40) -> list[str]:
    path = Path("screened_universe.csv")
    if not path.exists():
        return ["AAPL", "MSFT", "JPM", "XOM", "KO", "NVDA", "META", "GOOGL", "AMZN", "BRK-B"]
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return [str(s) for s in df[col].tolist()[:n]]


def _load_kor(n: int = 15) -> list[str]:
    path = Path("korean_universe.csv")
    if not path.exists():
        return ["005930.KS", "000660.KS", "035420.KS"]
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return [str(s) for s in df[col].tolist()[:n]]


def _load_hkg(n: int = 10) -> list[str]:
    path = Path("hongkong_universe.csv")
    if not path.exists():
        return ["0700.HK", "0005.HK", "9988.HK"]
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return [str(s) for s in df[col].tolist()[:n]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write FREE scan_results via save_scan_results (default: do not)",
    )
    parser.add_argument(
        "--usa",
        type=int,
        default=40,
        help="Number of USA symbols from screened_universe.csv (default: 40)",
    )
    parser.add_argument(
        "--kor",
        type=int,
        default=15,
        help="Number of KOR symbols (default: 15)",
    )
    parser.add_argument(
        "--hkg",
        type=int,
        default=10,
        help="Number of HKG symbols (default: 10)",
    )
    args = parser.parse_args(argv)

    usa = _load_usa(args.usa)
    kor = _load_kor(args.kor)
    hkg = _load_hkg(args.hkg)
    # USA first — same order as FREE merge typically exposes to rate limits first
    symbols = list(dict.fromkeys(usa + kor + hkg))
    cache_root = Path(tempfile.mkdtemp(prefix="momentum_usa_first_"))
    print(f"[usa-first] n={len(symbols)} (USA={len(usa)} KOR={len(kor)} HKG={len(hkg)})")
    print(f"[usa-first] cache={cache_root}")

    adapter = CachingMarketDataAdapter(
        YFinanceAdapter(threads=False),
        cache=DiskPriceCache(root=cache_root),
    )
    results = calculate_fms_for_batch(
        symbols,
        period_="1y",
        market_data=adapter,
        outer_batch_size=40,
    )
    if results.empty:
        print("[usa-first] FAIL: empty results")
        print(adapter.stats)
        return 1

    markets = Counter(classify(str(i)) for i in results.index)
    print(f"[usa-first] scored={len(results)} markets={dict(markets)} stats={adapter.stats}")

    kept = results[results["FMS"] >= DEFAULT_FMS_THRESHOLD]
    kept_m = Counter(classify(str(i)) for i in kept.index)
    print(f"[usa-first] FMS>={DEFAULT_FMS_THRESHOLD}: n={len(kept)} markets={dict(kept_m)}")

    ok = True
    for m in ("USA", "KOR", "HKG"):
        if markets.get(m, 0) < 1:
            print(f"[usa-first] FAIL: no scored rows for {m}")
            ok = False
        if kept_m.get(m, 0) < 1:
            print(f"[usa-first] FAIL: no FMS>={DEFAULT_FMS_THRESHOLD} rows for {m}")
            ok = False

    if not ok:
        return 1

    if args.save:
        success, msg, n = save_scan_results(
            results, fms_threshold=DEFAULT_FMS_THRESHOLD, mode=MODE_FREE
        )
        print(f"[usa-first] save: success={success} n={n} msg={msg}")
    else:
        print("[usa-first] skip save (pass --save to write scan_results/)")

    print("[usa-first] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
