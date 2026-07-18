# Work plan — 2026-07-18 — session summary (push-ready)

## Product version

**v4.4.5**

## What shipped today

| Ver | Change |
|-----|--------|
| 4.4.1 | `core/indicators` (`ema` / `returns_pct` / `r_squared_3m` / later `ytd`/`vol`); pre-filter **keep** |
| 4.4.2 | `core/tradeability` + `test_tradeability.py` |
| 4.4.3 | `core/fms` (`compute_fms_snapshot` / `momentum_now_and_delta`) |
| 4.4.4 | Recalib SSOT: `score_fms_from_feature_frame`; `f_current`/`f_proposed` → core; parity tests |
| 4.4.5 | Detail selectbox CSS/JS reverted to stock Streamlit |

## FMS / quant harness (how we validate)

| Layer | Asset | How to run |
|-------|--------|------------|
| Golden ranks / `-999` | `tests/unit/test_fms_scoring.py` + `tests/fixtures/*` | `python -m pytest` |
| Indicators | `tests/unit/test_indicators.py` | (pytest) |
| Tradeability TR/downside | `tests/unit/test_tradeability.py` | (pytest) |
| Recalib ↔ production | `tests/unit/test_fms_recalib_parity.py` | (pytest) |
| MarketDataPort | `tests/unit/test_market_data_port.py` | (pytest) |
| Batch I/O helpers | `test_yf_rate_limit_retry` / `test_finviz_ticker_normalize` | (pytest) |
| core no-network | `tests/contract/test_no_network_in_core.py` | (pytest) |
| Manual FMS table | `harness/run_fms_snapshot.py` | `python -m harness.run_fms_snapshot` |
| Prefilter evidence | `scripts/fixtures/prefilter_band_sample_fms.csv` | LIVE: `analyze_prefilter_impact.py` |

Production entrypoints (`app.py`, `run_scan_batch.py`) do **not** import `tests/` or `harness/`.
Scoring SSOT: `core/fms.py` (re-exported via `analysis_utils`).

## Push-prep cleanup

- Removed selectbox demo/CSS/JS (v4.4.5)
- Moved prefilter sample CSV → `scripts/fixtures/`
- Deduplicated `_mom_snapshot` local weight/helper forks → module-level `_P_*` / `_smoothstep` / `_z_peer` / `_z_ref`
- Docs synced to v4.4.5 (`HARNESS_RULES` §0, `TODO`, `CHANGELOG`, `.cursorrules`)

## Baseline verification (pre-push)

```text
python -m pytest
python -m harness.run_fms_snapshot
python -c "import app; import run_scan_batch; import analysis_utils; import config; from core.fms import compute_fms_snapshot"
```

## Next (not in this push)

- Optional: `get_filter_debug_info` → `core/tradeability`
- Tune `fms_score` → parameterized core / `calibration/`
- `universe_utils` / `watchlist_utils` → adapters
- Detail selectbox UX: wait for user feedback after stock behavior trial
