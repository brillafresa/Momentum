# 2026-08-05 — 배치↔UI FMS 경로 차이 실측 + v5 ops 수용

Status: `done` · product **v5.0.1**

## Decisions

- **배치 vs UI FMS 숫자 차이**: v5 절대점수에서는 peer/reference 영향 없음.
  동일 원본·연속 실행 시 경로(coverage 0.5 vs 0.9 등) ΔFMS는 ~0.01대·순위 불변.
  → **생산 패널 경로 통일하지 않음** (시차·날짜 경계가 큰 역전의 주원인).
- **사전필터**: v5 이후에도 Quarter/Half Up + SMA 유지. LIVE 재실측 없이 조이지/풀지 않음.
- **v5.0 FMS 추가 잔차/재피팅 라운드**: 운영 만족 → **열지 않음**.
  다음 세션 최우선 액션으로 잡지 말 것 (상세: `2026-08-02-fms-nonlinear-mc-refit.md`).

## Harness

- `harness/compare_batch_ui_fms.py` (+ `tests/unit/test_batch_ui_fms_paths.py`)
- LIVE 요약 CSV: `scan_results/batch_ui_fms_same_download.csv`,
  `scan_results/batch_ui_fms_mirror_io.csv` (로컬 산출물)

## Next session (if any)

중기 엔트리포인트/adapters 이전 또는 **신규 사용자 pain**만. FMS 잔차 라운드 기본값 아님.

## Follow-up same day

- [x] `run_scan_batch.py` relative-FMS abort/문구 → v5 절대 FMS (watchlist 비어도 스캔 계속)
- [x] `docs/README_BATCH.md` 동기화
