# TODO — Harness Engineering & Refactor Roadmap

> 세션 시작 시 `HARNESS_RULES.md` 다음으로 본 파일을 읽어 **직전 완료점 / 다음 액션**을 파악한다.  
> 최종 갱신: **2026-07-20** (KST) · 제품 버전 **v4.4.8**

상태 범례: `[x]` 완료 · `[ ]` 미착수 · `[~]` 진행 중

---

## 완료됨

### 2026-07-20 — v4.4.8 (UI 세부보기 + 운영 데이터 커밋 SSOT)

- [x] 세부보기 하단 Drawdown y축: 관심종목 전체 기준 고정 range (상단 Rebased 100과 동일 정책)
- [x] 좌측 [도구 및 도움말] `관심종목 초기화` 버튼 제거
- [x] 운영 데이터 3파일 항상 커밋 SSOT (`.cursorrules` / `HARNESS_RULES.md` / `CONTRIBUTING.md`)

### 2026-07-20 — v4.4.7 (FMS 6M→4M + Quarter/Half Up)

- [x] `R_6M`(126d) → `R_4M`(84d); 복리 게이트/quality/level + √t `gate_r4_w`
- [x] 사전필터 Finviz `Quarter Up` / `Half Up` + 로컬 `Perf > 0` (서버·로컬·문서 일치)
- [x] 계약: 사전필터 ≤ 로컬 (`test_prefilter_not_stricter_than_local.py` + `universe_utils` SSOT)
- [x] `test_fms_horizon_map.py`; 골든 순위 유지 확인

### 2026-07-20 — v4.4.6 (tune fms_score → core params)

- [x] `FmsScoreParams` / `production_fms_score_params()` (`core/fms.py`)
- [x] `score_fms_from_feature_frame(..., params=...)` Mapping/dataclass 주입
- [x] tune `fms_score` / `baseline_params` → core 위임 (공식 포크 삭제)
- [x] 계약 테스트 `tests/unit/test_fms_params.py`

### 2026-07-18 — v4.4.5 (세부보기 selectbox 순정 복원 + 푸시 전 정리)

- [x] 세부보기 selectbox CSS/JS 주입 제거 → Streamlit 기본 동작
- [x] `scripts/demo_focus_clear.py` 삭제 (사용 피드백 후 개선 여부 재결정)
- [x] 사전필터 실측 CSV → `scripts/fixtures/`; `core/fms` 헬퍼/가중치 중복 제거; 문서 §0 동기화

### 2026-07-18 — v4.4.4 ~ v4.4.1 / 2026-07-17 v4.4.0 / 2026-07-16

- [x] recalib SSOT, core fms/tradeability/indicators, MarketDataPort, 배치 복구, 하네스 부트스트랩
- [x] ~~사전 필터 Q+10/H+20 유지~~ → **v4.4.7에서 Q Up / H Up으로 정책 변경**

### 구축·유지 중인 검증 하네스 (요약)

| 자산 | 검증 내용 | 실행 |
|------|-----------|------|
| `tests/contract/test_prefilter_not_stricter_than_local.py` | Finviz Perf 사전필터 ≤ 로컬 | (pytest 포함) |
| `tests/unit/test_fms_horizon_map.py` | 복리/√t 매핑, R²=63d 불변 | (pytest 포함) |
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

상세 SSOT: [`HARNESS_RULES.md`](HARNESS_RULES.md) §0.

---

## 지금 당장 (Next — 우선순위 순)

- [ ] (선택) `get_filter_debug_info` → `core/tradeability.py` 동반 이전
- [ ] (선택) `fms_recalib_tune_vol_penalty` 단순화 본문 정리
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
