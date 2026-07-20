# TODO — Harness Engineering & Refactor Roadmap

> 세션 시작 시 `HARNESS_RULES.md` 다음으로 본 파일을 읽어 **직전 완료점 / 다음 액션**을 파악한다.  
> 최종 갱신: **2026-07-20** (KST) · 제품 버전 **v4.4.6**

상태 범례: `[x]` 완료 · `[ ]` 미착수 · `[~]` 진행 중

---

## 완료됨

### 2026-07-20 — v4.4.6 (tune fms_score → core params)

- [x] `FmsScoreParams` / `production_fms_score_params()` (`core/fms.py`)
- [x] `score_fms_from_feature_frame(..., params=...)` Mapping/dataclass 주입
- [x] tune `fms_score` / `baseline_params` → core 위임 (공식 포크 삭제)
- [x] 계약 테스트 `tests/unit/test_fms_params.py`

### 2026-07-18 — v4.4.5 (세부보기 selectbox 순정 복원 + 푸시 전 정리)

- [x] 세부보기 selectbox CSS/JS 주입 제거 → Streamlit 기본 동작
- [x] `scripts/demo_focus_clear.py` 삭제 (사용 피드백 후 개선 여부 재결정)
- [x] 사전필터 실측 CSV → `scripts/fixtures/`; `core/fms` 헬퍼/가중치 중복 제거; 문서 §0 동기화

### 2026-07-18 — v4.4.4 (recalib 공식 포크 제거)

- [x] `score_fms_from_feature_frame` (`core/fms.py`) — feature→score production 경로
- [x] `f_current` / `f_proposed` → core 호출 (독립 공식 삭제)
- [x] 계약 테스트 `tests/unit/test_fms_recalib_parity.py`
- [x] tune baseline → production (`f_current`); 탐색용 `fms_score*`는 오프라인 전용 명시

### 2026-07-18 — v4.4.3 (core/fms 이전)

- [x] `compute_fms_snapshot` / `momentum_now_and_delta` / `_mom_snapshot` → `core/fms.py` + 셔임
- [x] `ytd_return` / `last_vol_annualized` → `core/indicators.py` (FMS 의존 헬퍼)
- [x] `test_fms_scoring` 셔임 identity + 기존 골든/실격 회귀 유지

### 2026-07-18 — v4.4.2 (core/tradeability 이전)

- [x] `calculate_tradeability_filters` → `core/tradeability.py` + `analysis_utils` re-export 셔임
- [x] `tests/unit/test_tradeability.py` (CRASHY / OHLC 부족 / 짧은 시계열 / TR·하방 엣지 / 0 고저가 / 셔임)

### 2026-07-18 — v4.4.1 (core/indicators 이전 + 사전필터 유지 확정)

- [x] **사전 필터 유지 확정** (Q+10/H+20 등 현행 Finviz 조건 유지 — 당분간 재론 없음)
- [x] `ema` / `returns_pct` / `r_squared_3m` → `core/indicators.py` + `analysis_utils` re-export 셔임
- [x] `tests/unit/test_indicators.py` (EMA / returns / R² / glitch / 셔임 identity)

### 2026-07-17 — v4.4.0 (MarketDataPort + log 경고 수정)

- [x] `MarketDataPort` + `YFinanceAdapter` + `FixtureAdapter` (`adapters/market_data.py`)
- [x] `calculate_fms_for_batch` 다운로드/스코어 분리 (Port 주입, 기본 동작 불변)
- [x] 계약 테스트 `tests/unit/test_market_data_port.py` (fixture 배치 = 직접 스코어링 FMS 일치, no-network)
- [x] 배치 `invalid value encountered in log` 경고 수정 (음수/0 가격 가드 + 회귀 테스트)
- [x] 사전 필터 타이트함 실측 도구 `scripts/analyze_prefilter_impact.py` (LIVE, 수동 전용)
- [x] 세부보기 selectbox 붙여넣기 UX (CSS/JS) — **v4.4.5에서 순정 복원으로 철회**

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
| `tests/unit/test_fms_params.py` | params 기본값=production, 오버라이드, tune 위임 | (pytest 포함) |
| `tests/unit/test_fms_recalib_parity.py` | recalib feature→score = production snapshot FMS | (pytest 포함) |
| `tests/unit/test_fms_scoring.py` | 골든 순위, CRASHY→-999, OHLC 없음, NaN, yfinance 미호출 | `python -m pytest` |
| `tests/unit/test_tradeability.py` | True Range 실격·엣지 + analysis_utils 셔임 | (pytest 포함) |
| `tests/unit/test_indicators.py` | `core.indicators` EMA/returns/R² + analysis_utils 셔임 | (pytest 포함) |
| `tests/unit/test_yf_rate_limit_retry.py` | 429/`shared._ERRORS` 재시도 | (pytest 포함) |
| `tests/unit/test_finviz_ticker_normalize.py` | Finviz 티커 첫글자 중복 보정 | (pytest 포함) |
| `tests/unit/test_market_data_port.py` | FixtureAdapter 배치 = 직접 스코어링, no-network | (pytest 포함) |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | (pytest 포함) |
| `harness/run_fms_snapshot.py` | fixture → FMS 테이블 육안 확인 | `python -m harness.run_fms_snapshot` |
| `scripts/fixtures/generate_synthetic_panel.py` | seed=42 패널 재생성 | 필요 시만 |

상세 SSOT: [`HARNESS_RULES.md`](HARNESS_RULES.md) §0.

---

## 지금 당장 (Next — 우선순위 순)

- [ ] (선택) `get_filter_debug_info` → `core/tradeability.py` 동반 이전
- [ ] (선택) `fms_recalib_tune_vol_penalty.fms_score_with_vol_params` → core `vol_*` params 경로로 단순화 탐색 정리
- [ ] `fms_recalib_*.py` → `calibration/` 점진 이동

---

## 중기 (공식 드리프트 제거)

- [ ] `calibration_utils.py` → `calibration/session.py` + 셔임
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
