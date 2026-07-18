# Work plan — 2026-07-18 — core/indicators migration + pre-filter keep

## Product version

**v4.4.1**

## Completed

- Pre-filter decision locked: **keep current Finviz criteria** (no revisit for now)
- Migrated `ema` / `returns_pct` / `r_squared_3m` → `core/indicators.py`
- `analysis_utils` re-exports the same callables (identity shim for transitional callers)
- Added `tests/unit/test_indicators.py` (EMA, returns edge cases, R² rank/short/glitch, shim identity)
- Version sync: app / README / CHANGELOG / TODO / HARNESS_RULES / `.cursorrules`

## How related logic is validated

| Layer | Harness |
|-------|---------|
| Indicators | `tests/unit/test_indicators.py` |
| FMS ranks / -999 | `tests/unit/test_fms_scoring.py` + `harness/run_fms_snapshot` |
| core boundary | `tests/contract/test_no_network_in_core.py` |

## Baseline verification

```text
python -m pytest              # 22 passed
python -m harness.run_fms_snapshot
python -c "from core.indicators import ema; from analysis_utils import ema as e2; assert ema is e2"
```

## Next session

1. Read `HARNESS_RULES.md` + `TODO.md`; re-run pytest + harness smoke
2. Next TODO: `calculate_tradeability_filters` → `core/tradeability.py` + `test_tradeability.py`
3. Then: `compute_fms_snapshot` / `momentum_now_and_delta` → `core/fms.py`
