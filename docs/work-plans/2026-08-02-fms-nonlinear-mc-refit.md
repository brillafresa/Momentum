# 2026-08-02 — FMS 원점 재피팅 (비선형 + 몬테카를로)

Status: `promoted_to_production_v5.0.0` · **ops-accepted 2026-08-05**

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

## Ops follow-up (closed 2026-08-05)

- [x] ~~LIVE 실사용 후 잔차 모아 추가 라운드 여부 결정~~ → **추가 라운드 없음**
  - 사용자 운영 피드백: v5.0 FMS **대체로 만족** → 잔차 사냥/재피팅을 다음 세션 액션으로 잡지 않음
  - 알려진 캘리브 잔차 메모(HOMB/FBP under, MNPR over, CORT over-correction)는
    **이슈 티켓이 아니라 역사 메모**로만 유지. 신규 실사용 pain이 생기기 전에는 재개하지 않음
  - 재개 조건: 사용자가 구체 종목·순위 역전을 재피팅 후보로 명시할 때
    (`docs/FMS_RECALIBRATION_WORKFLOW.md` 절차)
