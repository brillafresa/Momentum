# Work Plan — UI 세부보기 Drawdown + 운영 데이터 커밋 SSOT

**날짜:** 2026-07-20 (KST)  
**버전:** v4.4.8

## 목표

1. 세부보기 하단 Drawdown 차트 y축을 상단 Rebased 100 차트와 같이 **관심종목 전체 기준**으로 통일
2. 실수 초기화 방지를 위해 `관심종목 초기화` 버튼 제거
3. 운영 데이터 3파일을 **항상 커밋에 포함**하도록 SSOT 문서화

## 구현

### 세부보기 Drawdown y-range

- `app.py` 세부보기 섹션에 `_global_drawdown_range(prices, period)` 추가
- 상단 `_global_rebased_log_range`와 동일하게:
  - 선택 차트 기간(1M/3M/6M/1Y/2Y) 윈도우 적용
  - `valid_for_scale`(FMS≠-999) 관심종목만 스케일 계산에 사용
  - 전 종목 Drawdown(%) concat 후 min/max + 5% 패딩 → `fig_dd.update_yaxes(range=...)`

### UI

- 좌측 [도구 및 도움말] `🔄 관심종목 초기화` 버튼 및 저장 로직 삭제
- `DEFAULT_*_SYMBOLS`는 세션 최초 로드·모드 전환 폴백으로 유지

### 운영 데이터 커밋 SSOT

- `.cursorrules`, `HARNESS_RULES.md`, `docs/CONTRIBUTING.md`에 명시:
  - `watchlist_free.csv`, `watchlist_irp.csv`, `screened_universe.csv`
  - 코드 전용 커밋에서도 제외하지 않음

## 검증

```bash
python -m pytest
python -m harness.run_fms_snapshot
python -m py_compile app.py run_scan_batch.py
```

- pytest 52 passed (2026-07-20)
- harness 골든 순위: TREND_UP > MILD_UP > FLAT > CRASHY(-999)

## 프로덕션-하네스 경계

- `app.py` / `run_scan_batch.py`는 `tests/`·fixture 경로 미import (기존 유지)
- `config.py`는 운영 런타임 설정만; Mock/fixture는 `tests/` · `harness/` · `scripts/`
