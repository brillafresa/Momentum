# 2026-08-08 — 캐시 period 불일치 (v5.0.5)

Status: `done` · product **v5.0.5**

## Pain

신규 종목 탐색에서 ITGR 등을 관심종목에 추가하면
「다음 종목은 데이터 부족으로 표시되지 않습니다: ITGR」 — 실제 Yahoo 데이터는 있음.

## Root cause

1. 배치는 `period=1y`로 `cache/market_data` write-through.
2. UI는 `calculate_minimum_data_period` → 보통 `2y` 요청.
3. v5.0.4 warm HIT는 **날짜만** 비교 → 1y 시계열 HIT.
4. 이미 2y인 다른 관심종목과 concat 시 ITGR leading NaN ≈50% →
   `harmonize_calendar(coverage=0.5)` 탈락.

## Fix

`cache_covers_request(cached, requested_period, cached_period)`:
- meta period rank가 요청보다 짧거나
- non-null bars &lt; period floor (2y→360 등)
→ **period miss** (probe 없이 full 재다운로드 + write-through).

## Harness

- `tests/unit/test_price_cache_freshness.py`
  - `test_cache_covers_request_period_rank_and_bars`
  - `test_caching_adapter_period_miss_refreshes_for_longer_request`
- UI: ITGR 1y 캐시 리셋 → watchlist 포함 로드 → 경고 없음 · 가속보드/비교차트 표시 ·
  캐시 `2y`/501 bars

## Related

- `docs/work-plans/2026-08-08-cache-probe-warm-only.md` (v5.0.4)
