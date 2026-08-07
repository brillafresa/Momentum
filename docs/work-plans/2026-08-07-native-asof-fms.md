# 2026-08-07 — 종목별 native as-of FMS (v5.0.2)

Status: `done` · product **v5.0.2**

## Pain

- UI FREE watchlist(ITGR) FMS ≈ **10.24** vs 배치 단건 ≈ **5.15**
- 원인: 다국가 concat 후 `harmonize_calendar` blanket ffill이 다른 시장만 열린
  trailing day에 가짜 평봉을 만들어 `SEG_*` 창이 한 칸 밀림
- 방향 무관: KR 선행뿐 아니라 US/HK 선행·향후 시장에도 동일

## Fix

- `core.indicators.harmonize_calendar`: 컬럼별 `last_valid_index` 너머 NaN 복원
- `returns_pct` / `last_vol_annualized` / `ytd_return`: 컬럼별 last valid
- `analysis_utils` re-export · `app.py` 로컬 harmonize 복제 제거

## Harness

- `tests/unit/test_native_asof_calendar.py`
- `tests/unit/test_indicators.py` (as-of returns + shim)
- `tests/unit/test_batch_ui_fms_paths.py` (trailing as-of 보존)

## Verification

- `python -m pytest` EXIT=0
- `python -m pytest tests/unit/test_native_asof_calendar.py -q`
- `python -m harness.run_fms_snapshot` 골든 순위 유지
- `python -m harness.compare_batch_ui_fms --offline`
- app / `run_scan_batch` import 스모크 OK (tests/harness 미import)
- `config.py`: production FMS_FORMULA / DEFAULT_FMS_THRESHOLD만 유지 (fixture 경로 없음)
