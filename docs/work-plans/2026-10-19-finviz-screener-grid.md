# 2026-10-19 — Finviz 스크리너 `fv-table` 파서

Status: `done` · product **v5.0.13**

## Pain / Intent

`run_batch_manual.bat` FREE 모드가 유니버스 갱신에서
`스크리닝 결과가 비어있습니다. (symbols: 0)` 로 종료.
필터는 그대로이고, 같은 요청의 실측 총건수는 약 470개.

## Cause

Finviz가 `table.screener_table` + `select#pageSelect` + `.count-text` 총건수를
`div.fv-table` 그리드로 교체했다.

- `.count-text`는 이제 `Filters: sh_price:o10 ...` 만 담는다.
- 총건수는 `.fv-table_toolbar-count` (`#1 / N Total`).
- `#pageSelect`가 없으면 `finviz_effective_page_count`가 0 → 빈 DataFrame.
- 티커 표시 텍스트는 로고 이니셜과 합쳐 `A A`. 심볼은 `data-ticker`.
- 페이지 인자 `r`은 그대로 동작.
- `finvizfinance` 1.5.0 소스도 `screener_table` / `pageSelect`를 본다.
  패키지 업그레이드만으로는 이 경로가 복구되지 않는다.

## Changes (final)

### Production (`universe_utils.py`)

- `parse_finviz_total_count`: toolbar 총건수 우선, 레거시 `.count-text` 유지
- `pageSelect` 부재 시 기존 식 `ceil(total / 20)` 사용
- `parse_finviz_grid_page`: `.fv-table_row` / `.fv-table_cell`
  - 티커 = `data-ticker`
  - 긴 헤더만 사용 (`Market Cap`, `Change %` → `Change`)
  - `NUMBER_COL`은 `number_covert` (Price float, Change 비율)
- `screener_table` HTML은 `overview._parse_table` 유지
- 서버 필터·로컬 후필터·`r` 페이지네이션·partial fallback 불변
- Overview에 `Avg Volume` / `Perf Quarter` / `SMA50` 열이 없는 점은 변경 전과 같다.
  그 조건은 Finviz `set_filter`가 이미 적용한다.

### Boundary

- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- `screened_universe.csv`는 이 수정만으로 다시 쓰지 않는다

## FMS

production FMS (`alive_pullback`)와 SMA5 거래적합성은 이 작업에서 변경하지 않는다.
검증은 기존 `test_fms_scoring` · `test_fms_alive_pullback_production` · `test_tradeability`가 유지한다.

## Harness

| Asset | What it locks |
|-------|----------------|
| `tests/unit/test_finviz_screener_pagination.py` | toolbar 총건수 · pageSelect 없는 24페이지 식 · 그리드 2페이지 · `data-ticker` · Price/Change · 레거시 partial |
| `tests/contract/test_prefilter_not_stricter_than_local.py` | 필터 SSOT 불변 |

```bash
python -m pytest tests/unit/test_finviz_screener_pagination.py tests/contract/test_prefilter_not_stricter_than_local.py -q
```

Live (1 page, CSV 미기록): total 473, pages 24, rows 20, tickers `A`/`AAPL`, Price min 10.5.

## Other Finviz callers (same day)

| Caller | HTML? | Result |
|--------|--------|--------|
| UI 가속보드·비교차트·세부보기 | yfinance | 개편 무관 |
| UI 종목 추가/삭제 | watchlist CSV | 개편 무관 |
| UI 종목명 | 캐시 · 유니버스 CSV · yfinance | 개편 무관 |
| UI 신규 종목 탐색 | `scan_results/latest_scan_results_*.csv` | 개편 무관. 「지금 배치 강제 실행」은 수정된 배치와 동일 경로 |
| `scripts/analyze_prefilter_impact.py` 건수 | 총건수 문자열 | 현 HTML에서도 `#1 / N Total`이 있어 동작. 파서는 `parse_finviz_total_count`로 통일 |
| 같은 스크립트 `--sample` | `Performance.screener_view` | `pageSelect` 부재로 0건. `finviz_screener_view_resilient`로 교체 |
