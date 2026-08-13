# 2026-08-13 — 최근 IPO native-span coverage 오탈락 (v5.0.6)

Status: `done` · product **v5.0.6**

## Pain

UI에서 LBRX, VIA를 관심종목에 추가하면
「다음 종목은 데이터 부족으로 표시되지 않습니다: LBRX, VIA」.
Yahoo에는 데이터가 있다.

## Root cause

1. LBRX IPO 2025-09-11, VIA IPO 2025-09-12 → 오늘(2026-08-13) 기준 ~11개월(~230 B-day).
2. UI는 기존 관심종목 때문에 `min_data_period=2y` (~504 B-day) 패널을 concat.
3. `harmonize_calendar` coverage = `count / len(union index)` → ~230/504 ≈ 0.46 < 0.5.
4. 다운로드/캐시 miss가 아님 (v5.0.5 ITGR period HIT와 별개). 상장 전 leading NaN을
   “결측”으로 본 필터 버그.

## Fix

`core/indicators.py` `_native_span_coverage`: 분모를 컬럼
`[first_valid_index, last_valid_index]`로 제한. 상장 전 leading · native as-of 이후
trailing NaN은 비율에 넣지 않음. all-NaN 컬럼은 계속 제외.

## Harness

- `tests/unit/test_native_asof_calendar.py`
  - `test_harmonize_keeps_recent_ipo_on_long_peer_calendar`
  - `test_fms_unchanged_when_recent_ipo_column_concatenated`
  - `test_harmonize_drops_all_nan_column`
- `tests/unit/test_batch_ui_fms_paths.py`
  - `test_late_listing_kept_by_ui_and_batch_coverage`
- 전체 `python -m pytest`

## Related

- `docs/work-plans/2026-08-08-cache-period-mismatch.md` (v5.0.5, 다른 원인)

## Push-prep (2026-08-13)

- 운영 코드에 티커 하드코딩 없음 (`app.py` coverage 주석은 native-span 의미만)
- Mock/회귀는 `tests/unit/test_native_asof_calendar.py` · `test_batch_ui_fms_paths.py`
- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- SSOT: `HARNESS_RULES.md` §0 / §3.6 · `.cursorrules` 다국가 캘린더 절
