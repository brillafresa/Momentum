# Work plan — 2026-07-16 — Harness bootstrap + pre-push cleanup

## Product version

**v4.3.0**

## Completed

- Harness Engineering plan phases 1–3 (rules, first FMS pytest harness, fixtures, CLI runner)
- Session bootstrap: `HARNESS_RULES.md`, `TODO.md`, `docs/`, `scripts/`
- Moved workflow/deployment/contributing/batch guides under `docs/`
- Pre-push cleanup:
  - Removed commented debug prints in `watchlist_utils.py`
  - `core/__init__.py` no longer re-exports `analysis_utils` (keeps pure package boundary)
  - Documented harness modules with purpose + run instructions
  - Added `scripts/fixtures/generate_synthetic_panel.py`
  - Synced version to v4.3.0 across app / README / CHANGELOG / `.cursorrules`

## How FMS logic is validated

1. Synthetic panel (TREND_UP / MILD_UP / FLAT / CRASHY) + OHLC → `momentum_now_and_delta`
2. Assert golden order and CRASHY FMS == -999; patch `yfinance.download` to fail if called
3. Manual smoke: `python -m harness.run_fms_snapshot`

## Baseline verification (pre-push)

```text
python -m pytest
python -m harness.run_fms_snapshot
python -c "import app; import run_scan_batch; import analysis_utils; import config"
```

## Next session

1. Read `HARNESS_RULES.md` + `TODO.md`
2. Re-run pytest + harness smoke
3. Next TODO: MarketDataPort / download-score split / `core/indicators.py`
