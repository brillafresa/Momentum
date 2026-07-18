# Work plan — 2026-07-18 — recalib formula fork removal

## Product version

**v4.4.4**

## Completed

- Added `core.fms.score_fms_from_feature_frame` (production reference-panel formula
  on a precomputed feature table)
- Replaced `fms_recalib_evaluate_formulas.f_current` / `f_proposed` with thin
  wrappers around core (no independent formula body)
- Contract tests: `tests/unit/test_fms_recalib_parity.py`
  - feature-frame score == `compute_fms_snapshot(..., reference_prices_krw=prices)`
  - `f_current` / `f_proposed` delegate to core
- Tune scripts: baseline metrics now use production `f_current`; parameterized
  `fms_score` / `fms_score_with_vol_params` marked search-only

## How related logic is validated

| Layer | Harness |
|-------|---------|
| Production ↔ recalib parity | `test_fms_recalib_parity.py` |
| Golden ranks / -999 | `test_fms_scoring.py` + harness CLI |
| core boundary | `test_no_network_in_core.py` |

## Baseline verification

```text
python -m pytest              # 34 passed
python -m harness.run_fms_snapshot
```

## Next

1. Optional: `get_filter_debug_info` → `core/tradeability`
2. Integrate/parameterize tune `fms_score` against core, or move under `calibration/`
3. Mid-term: `universe_utils` / `watchlist_utils` → adapters
