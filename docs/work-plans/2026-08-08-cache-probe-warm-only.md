# 2026-08-08 — 캐시 warm-only probe (v5.0.4)

Status: `done` · product **v5.0.4**

## Pain

- 운영 클론에서 FREE 배치 후 **신규 종목 탐색에 USA가 없고 KR/HK만** 남는 현상 보고.
- v5.0.3 `CachingMarketDataAdapter`가 **cold 심볼까지** 매 요청 5d last-bar probe를 수행 →
  Yahoo 호출 ≈2–4배(가격 probe+full + OHLC probe+full).
- FREE 유니버스는 USA가 앞쪽 → 초반 outer chunk가 rate-limit에 실패하고 뒤쪽 KR/HK만 성공.

## Fix

- Cold (디스크 캐시 없음): probe 생략 → full period download 1회만.
- Warm (캐시 있음): 기존처럼 5d probe → 같은 calendar date면 HIT, 신규일이면 refresh.
- `stats["cold_misses"]`로 관측 가능하게 유지 (운영 미노출; adapter 내부).

## Harness (FMS/배치 경로 — 캐시 I/O)

| 자산 | 목적 | 실행 |
|------|------|------|
| `tests/unit/test_price_cache_freshness.py` | cold → probe=0 · warm → HIT | `pytest … -q` |
| `harness/smoke_multi_market_batch.py` | LIVE 소규모 USA+KOR+HKG | `python -m harness.smoke_multi_market_batch` |
| `harness/smoke_usa_first_batch.py` | LIVE USA-선행 중규모 | `python -m harness.smoke_usa_first_batch` |

운영 `app.py` / `run_scan_batch.py`는 `harness/`·`tests/`를 import하지 않음.
LIVE smoke는 기본으로 **임시 캐시 루트**를 쓰며 production `cache/`·`scan_results/`를 덮어쓰지 않음
(`--save` / `--tmpdir`만 명시 시 예외).

## Verification

1. Unit: cold path `probes==0`, warm path HIT.
2. LIVE multi-market smoke: 3시장 ≥1 row.
3. LIVE USA-first smoke: USA/KOR/HKG 모두 FMS≥0 행 존재.
4. 로컬 운영형 FREE 풀 배치: Finviz 갱신 + 전 유니버스 →
   `latest_scan_results_free.csv`에 USA 다수 + UI 신규탐색 1페이지 USA 티커 확인.
5. 푸시 전: 전체 `pytest` · app/`run_scan_batch` import 스모크.

## Related

- 선행: `docs/work-plans/2026-08-08-detail-view-cache.md` (v5.0.3)
- SSOT: `HARNESS_RULES.md` §0 · `CHANGELOG.md` [5.0.4]
