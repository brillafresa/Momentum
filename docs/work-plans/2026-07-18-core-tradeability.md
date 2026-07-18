# Work plan — 2026-07-18 — core/tradeability migration

## Product version

**v4.4.2**

## Completed

- Migrated `calculate_tradeability_filters` → `core/tradeability.py`
- `analysis_utils` re-exports the same callable (identity shim)
- Added `tests/unit/test_tradeability.py` (fixture CRASHY, missing OHLC, short
  history, extreme TR, repeated downside, zero H/L glitch, flat columns, shim)
- Left `get_filter_debug_info` in `analysis_utils` (optional follow-up)
- Version sync: app / README / CHANGELOG / TODO / HARNESS_RULES / `.cursorrules`

## How related logic is validated

| Layer | Harness |
|-------|---------|
| Tradeability | `tests/unit/test_tradeability.py` |
| FMS -999 path | `tests/unit/test_fms_scoring.py` + `harness/run_fms_snapshot` |
| core boundary | `tests/contract/test_no_network_in_core.py` |

## Baseline verification

```text
python -m pytest              # 30 passed
python -m harness.run_fms_snapshot
```

## Next session / immediate next

1. `compute_fms_snapshot` / `momentum_now_and_delta` (+ `_mom_snapshot`) → `core/fms.py`
2. Optional: move `get_filter_debug_info` beside tradeability
