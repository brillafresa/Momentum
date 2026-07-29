# 2026-07-29 — FMS 원점 재피팅

## 상태

- 제품 버전: v4.6.0
- 후보 상태: `promoted_to_production`
- production FMS 변경: 승인 sparse-linear 모델 승격
- 결정: 사용자 승인 완료

## 이번 완료점

1. JSON `saved_at` 기준 최신 완료 재보정 세션 하나만 선택했다.
2. ranking/snapshot hash와 development/audit split을 manifest로 고정했다.
3. production FMS는 benchmark로만 사용하고 후보 점수를 0에서 시작했다.
4. sparse linear, monotone GAM, 제한 상호작용 후보를 비교했다.
5. nested holdout, bootstrap, LOO, review label 전 변형을 검증했다.
6. development 오분류 차트를 검토하고 일반화 가능한 피처를 재검토했다.
7. 후보 보고 단계에서 중지한 뒤 사용자 승인 후 v4.6.0 production으로 승격했다.

## 주요 자산

- 절차 SSOT: `docs/FMS_RECALIBRATION_WORKFLOW.md`
- 피처: `core/fms_features.py`
- manifest: `calibration/manifest.py`
- 지표: `calibration/ranking_metrics.py`
- 원점 피팅: `calibration/fms_recalib_refit.py`
- 잔차 차트: `calibration/fms_recalib_plot_residuals.py`
- 후보 보고: `fms_recalib_scratch_candidate.json`

## 다음 액션

- [x] production scorer / parity / 골든 / 버전 / README / CHANGELOG / TODO / HARNESS / UI 동기화
- [x] 푸시 전 dead-code·harness 경계·문서 SSOT 정리 및 pytest/import 스모크
