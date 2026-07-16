# Work plan — 2026-07-16 — Batch recovery + dividend policy (push-ready)

## Product version

**v4.3.1**

## Completed

- Diagnosed batch failure:
  1. yfinance swallows rate limits into `shared._ERRORS` → retries never fired
  2. Finviz Overview duplicates first ticker char → mass "no data"
  3. Finviz `set_filter` never called → ~7k dump
- Hardened downloads (backoff, chunk sleep, outer batches)
- `normalize_finviz_tickers` + unit tests
- FREE/IRP batches completed; `latest_scan_results_*.csv` refreshed
- Dividend policy confirmed: Adj Close returns/FMS; raw OHLC tradeability
- Pre-push cleanup: harness docs/headers, unused import, SSOT sync

## How related logic is validated

| Layer | Harness |
|-------|---------|
| FMS ranks / -999 | `tests/unit/test_fms_scoring.py` + fixtures + `harness/run_fms_snapshot` |
| Yahoo 429 retry | `tests/unit/test_yf_rate_limit_retry.py` (mocked) |
| Finviz ticker fix | `tests/unit/test_finviz_ticker_normalize.py` |

## Baseline verification (pre-push)

```text
python -m pytest
python -m harness.run_fms_snapshot
python -c "import app; import run_scan_batch; import analysis_utils; import universe_utils; import config"
```

## Next session

1. Read `HARNESS_RULES.md` + `TODO.md`
2. Re-run pytest + harness smoke
3. Next TODO: MarketDataPort / download-score Port split / `core/indicators.py`
