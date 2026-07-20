# -*- coding: utf-8 -*-
"""
Batch pre-filter tightness evidence collector.

**LIVE API**: This script calls finviz.com (count pages only, one request per
filter combo). It is an offline analysis aid — never imported by production
entrypoints. Run manually from repo root:

    python scripts/analyze_prefilter_impact.py

Purpose
-------
Quantify how many US symbols are discarded by the current server-side Finviz
pre-filter (Quarter Up / Half Up) versus tighter historical variants, to judge
whether stocks with potential FMS > 0 are being pre-filtered away
(FMS trend gates open at R_3M≈5% / R_4M≈5.3%, i.e. related to but distinct from
the pre-filter). Historical sample CSV under scripts/fixtures/ was captured
under the older Q+10%/H+20% policy (2026-07-17).
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests
from finvizfinance.screener.overview import Overview

# Allow running as `python scripts/analyze_prefilter_impact.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_FILTERS = {
    'Price': 'Over $10',
    'Average Volume': 'Over 300K',
    '50-Day Simple Moving Average': 'Price above SMA50',
    '200-Day Simple Moving Average': 'Price above SMA200',
}

COMBOS = {
    'current  (Quarter Up, Half Up)    ': {'Performance': 'Quarter Up', 'Performance 2': 'Half Up'},
    'legacy   (Quarter +10%, Half +20%)': {'Performance': 'Quarter +10%', 'Performance 2': 'Half +20%'},
    'mid      (Quarter Up,   Half +10%)': {'Performance': 'Quarter Up', 'Performance 2': 'Half +10%'},
    'sma-only (no performance filter)  ': {},
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def fetch_total(filters_dict: dict) -> str:
    """Fetch only the first screener page and parse the 'Total' counter.

    finvizfinance keeps filters in ``request_params`` (not in ``.url``), so we
    must pass them as query params explicitly.
    """
    view = Overview()
    view.set_filter(filters_dict=filters_dict)
    html = requests.get(view.url, params=view.request_params, headers=HEADERS, timeout=30).text
    m = re.search(r'#\d+\s*/\s*([\d,]+)\s*Total', html)
    if m:
        return m.group(1)
    return f'count not found (html {len(html)} chars)'


def sample_borderline_fms(sample_size: int) -> int:
    """Score a random relaxed-universe sample with the production FMS path.

    Samples from the relaxed pre-filter set (Quarter Up / Half Up), scores it
    exactly like ``run_scan_batch`` (watchlist reference distribution), then
    splits post-hoc by the *computed* R_3M/R_4M into:

    - borderline band: fails Up policy (R_3M<=0 or R_4M<=0)
    - passing set: satisfies Up policy

    This answers: "does the current pre-filter discard FMS>0 candidates?"
    (Post-hoc split avoids trusting Finviz perf columns, whose units drift.)
    """
    from finvizfinance.screener.performance import Performance

    print('[Prefilter] Fetching relaxed universe (Quarter Up / Half Up)...')
    view = Performance()
    view.set_filter(filters_dict=dict(BASE_FILTERS, **{
        'Performance': 'Quarter Up', 'Performance 2': 'Half Up',
    }))
    df = view.screener_view(verbose=0)
    if df is None or df.empty:
        print('[Prefilter] No screener rows returned; aborting sample stage.')
        return 1
    print(f'[Prefilter] Relaxed universe rows: {len(df)}')

    from universe_utils import normalize_finviz_tickers
    tickers = normalize_finviz_tickers(df['Ticker'].astype(str).tolist())
    sample_tickers = (
        pd.Series(tickers).drop_duplicates().sample(
            n=min(sample_size, len(tickers)), random_state=42
        ).tolist()
    )
    print(f'[Prefilter] Sampled {len(sample_tickers)} tickers: {sample_tickers}')

    from watchlist_utils import load_watchlist, MODE_FREE
    from analysis_utils import build_prices_krw_from_symbols, calculate_fms_for_batch

    watchlist = load_watchlist([], mode=MODE_FREE)
    print(f'[Prefilter] Building watchlist reference panel ({len(watchlist)} symbols)...')
    ref_prices = build_prices_krw_from_symbols('1Y', watchlist)
    if ref_prices.empty:
        print('[Prefilter] Reference panel empty; aborting.')
        return 1

    print('[Prefilter] Scoring sample with production batch path...')
    results = calculate_fms_for_batch(sample_tickers, reference_prices_krw=ref_prices)
    if results.empty:
        print('[Prefilter] No scores produced.')
        return 1

    scored = results[results['FMS'] > -900].copy()  # exclude -999 disqualifications
    n_disq = len(results) - len(scored)
    r4col = 'R_4M' if 'R_4M' in scored.columns else 'R_6M'
    border = scored[(scored['R_3M'] <= 0.0) | (scored[r4col] <= 0.0)]
    passing = scored[(scored['R_3M'] > 0.0) & (scored[r4col] > 0.0)]

    print()
    print(f'[Prefilter] scored={len(scored)}  disqualified(-999)={n_disq}')
    print(f'[Prefilter] borderline band (current filter would discard): {len(border)}')
    for thr in (0.0, 0.5, 1.0):
        print(f'  band FMS > {thr}: {int((border["FMS"] > thr).sum())} / {len(border)}')
    if not border.empty:
        print(f'  band FMS max: {border["FMS"].max():.3f}  median: {border["FMS"].median():.3f}')
    print(f'[Prefilter] passing set (kept by current filter): {len(passing)}')
    for thr in (0.0, 0.5, 1.0):
        print(f'  pass FMS > {thr}: {int((passing["FMS"] > thr).sum())} / {len(passing)}')

    out_path = 'scripts/fixtures/prefilter_band_sample_fms.csv'
    results.to_csv(out_path, encoding='utf-8-sig')
    print(f'[Prefilter] Full sample results saved: {out_path}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Batch pre-filter tightness evidence (LIVE Finviz/Yahoo).')
    parser.add_argument('--sample', type=int, default=0,
                        help='Also score N borderline-band symbols with production FMS (0 = counts only)')
    args = parser.parse_args()

    print('[Prefilter] Finviz universe size per filter combo (count page only):')
    for name, perf in COMBOS.items():
        filters = dict(BASE_FILTERS, **perf)
        try:
            total = fetch_total(filters)
        except Exception as e:
            total = f'error: {e}'
        print(f'  {name}: {total}')
        time.sleep(2.0)

    if args.sample > 0:
        return sample_borderline_fms(args.sample)
    return 0


if __name__ == '__main__':
    sys.exit(main())
