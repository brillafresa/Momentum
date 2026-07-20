# Work plan — 2026-07-20 — FMS 6M→4M + Quarter/Half Up

## Product version

**v4.4.7**

## Completed

- Long-horizon FMS axis: `R_6M`(126d) → `R_4M`(84d)
- Thresholds: compound map for gate center / quality / `level_r4_hi`; √t for `gate_r4_w`; `w_r4` unchanged
- Helpers: `horizon_return_map`, `gate_width_scale` + `test_fms_horizon_map.py`
- Prefilter: Finviz `Quarter Up` + `Half Up`; local `Perf Quarter/Half > 0`
- Contract harness: Finviz Perf exclusive floor ≤ local (`test_prefilter_not_stricter_than_local`)
- Docs/UI/config/recalib builders aligned; golden rank order unchanged on synthetic fixture

## Verification

```text
python -m pytest
python -m harness.run_fms_snapshot
```

## Notes

- Chart UI period label `"6M"` is unrelated to the FMS `R_4M` feature.
- `prefilter_band_sample_fms.csv` remains a Q+10/H+20-era evidence snapshot.
