# 2026-08-02 — FMS 원점 재피팅 (비선형 + 몬테카를로)

Status: `promoted_to_production_v5.0.0`

## Result (promoted)

- Family: **`alive_pullback`**
- Full: inv **0.146** / Spearman **0.877** / pair-delta **23.4**
  vs then-production 0.274 / 0.566 / 41.2
- Audit inv tied / rho better than then-production; label variants 32/32
- Production SSOT: `core/fms_features.py`
  (`PRODUCTION_ALIVE_PULLBACK_PARAMS`, `score_alive_pullback_from_params`)
- Evidence: `fms_recalib_scratch_candidate.json` status=`promoted_to_production_v5.0.0`

## Harness built this session

- SEG_* non-overlapping + residual features (`MID_DIP_RECOVERY`, `STALE_AFTER_RUN`, …)
- `calibration/nonlinear_formulas.py` + `fms_recalib_nonlinear_mc.py`
- `tests/unit/test_nonlinear_mc_features.py`
- `tests/unit/test_fms_alive_pullback_production.py`
- residual plot schema fix for symbol-gap CSV

## Next (ops feedback)

- [ ] LIVE 앱/배치로 실사용 후 잔차 이슈 모아서 추가 라운드 여부 결정
  (알려진 잔차: HOMB/FBP under, MNPR over, CORT over-correction)
