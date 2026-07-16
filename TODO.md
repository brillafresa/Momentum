# TODO — Harness Engineering & Refactor Roadmap

> 세션 시작 시 `HARNESS_RULES.md` 다음으로 본 파일을 읽어 **직전 완료점 / 다음 액션**을 파악한다.  
> 최종 갱신: **2026-07-16** (KST) · 제품 버전 **v4.3.1**

상태 범례: `[x]` 완료 · `[ ]` 미착수 · `[~]` 진행 중

---

## 완료됨

### 2026-07-16 — v4.3.1 (배치 복구 + 배당 정책 확정)

- [x] Yahoo 레이트리밋 재시도 복구 (`shared._ERRORS` 감지·백오프·outer batch)
- [x] Finviz `set_filter` 적용 + 티커 첫 글자 중복 정규화
- [x] FREE/IRP 배치 완료·`latest_scan_results_*.csv` 갱신
- [x] 배당 정책 확정: Adj Close 수익/FMS + raw OHLC 필터
- [x] 하네스: `test_yf_rate_limit_retry.py`, `test_finviz_ticker_normalize.py`

### 2026-07-16 — v4.3.0 (하네스 부트스트랩)

- [x] 하네스 도입·`compute_fms_snapshot`·FMS pytest/fixture/CLI
- [x] `docs/` 이관·`core`/`adapters` 스캐폴딩·버전 동기화

### 구축·유지 중인 검증 하네스 (요약)

| 자산 | 검증 내용 | 실행 |
|------|-----------|------|
| `tests/unit/test_fms_scoring.py` | 골든 순위, CRASHY→-999, OHLC 없음, NaN, yfinance 미호출 | `python -m pytest` |
| `tests/unit/test_yf_rate_limit_retry.py` | 429/`shared._ERRORS` 재시도 | (pytest 포함) |
| `tests/unit/test_finviz_ticker_normalize.py` | Finviz 티커 첫글자 중복 보정 | (pytest 포함) |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | (pytest 포함) |
| `harness/run_fms_snapshot.py` | fixture → FMS 테이블 육안 확인 | `python -m harness.run_fms_snapshot` |
| `scripts/fixtures/generate_synthetic_panel.py` | seed=42 패널 재생성 | 필요 시만 |

상세 SSOT: [`HARNESS_RULES.md`](HARNESS_RULES.md) §0.

---

## 지금 당장 (Next — 우선순위 순)

- [ ] `MarketDataPort` + `YFinanceAdapter` + `FixtureAdapter` (`adapters/market_data.py`)
- [ ] `calculate_fms_for_batch`에서 **다운로드 / 스코어 분리** (Port 주입; outer batch는 완료)
- [ ] `ema` / `returns_pct` / `r_squared_3m` → `core/indicators.py` + re-export 셔임
- [ ] `calculate_tradeability_filters` → `core/tradeability.py`
- [ ] `compute_fms_snapshot` / `momentum_now_and_delta` → `core/fms.py`
- [ ] `tests/unit/test_indicators.py` · `test_tradeability.py` 추가

---

## 중기 (공식 드리프트 제거)

- [ ] `fms_recalib_evaluate_formulas.f_current` / `f_proposed` → core FMS 호출로 교체
- [ ] tune 스크립트 독립 `fms_score` 제거·통합
- [ ] `fms_recalib_*.py` → `calibration/` 점진 이동
- [ ] `calibration_utils.py` → `calibration/session.py` + 셔임
- [ ] production vs recalib **계약 테스트** (동일 fixture → 동일 FMS)
- [ ] (선택) UI에 배당 기여분(가격수익 vs 총수익) 분해 표시

---

## 중기 (엔트리포인트 정리)

- [ ] `app.py` 중복 `download_*` / indicator 제거 → adapters + core
- [ ] `universe_utils` / `watchlist_utils` → `adapters/` + 셔임
- [ ] 배치 fixture 스모크 (네트워크 없이 스코어만)

---

## 명시적 비범위

- 전 파일 big-bang 이동
- Streamlit UI 리디자인
- 라이브 API E2E를 CI 필수화
- OHLC `auto_adjust=True` 전환 (v4.3.1에서 기각)

---

## 빠른 명령

```bash
python -m pytest
python -m harness.run_fms_snapshot
```
