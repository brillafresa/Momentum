# Work plan — 2026-07-20 — tune fms_score → core params

## Product version

**v4.4.6**

## Completed

- Introduced `FmsScoreParams` + `production_fms_score_params()` in `core/fms.py`
- `score_fms_from_feature_frame(..., params=...)` accepts dataclass or Mapping
- `_mom_snapshot` reference/peer paths read shared production params (peer primary weights remain `_PEER_W_*`)
- Replaced tune script formula fork:
  - `fms_score` → `score_fms_from_feature_frame(df, params=p)`
  - `baseline_params()` → `asdict(production_fms_score_params())`
- Contract tests: `tests/unit/test_fms_params.py`

## How related logic is validated

| Layer | Harness |
|-------|---------|
| Params default = production | `test_fms_params.py` |
| Params override changes score | `test_fms_params.py` |
| Tune delegates to core | `test_fms_params.py` |
| Feature ↔ snapshot parity | `test_fms_recalib_parity.py` |
| Golden ranks / -999 | `test_fms_scoring.py` + harness CLI |

## Baseline verification

```text
python -m pytest              # 41 passed
python -m harness.run_fms_snapshot
# TREND_UP > MILD_UP > FLAT > CRASHY(-999) unchanged
```

## Next

1. Optional: `get_filter_debug_info` → `core/tradeability`
2. Optional: vol-penalty tune simplified body → core `vol_*` params
3. `fms_recalib_*.py` → `calibration/` gradual move
