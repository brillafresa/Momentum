# TODO — Harness Engineering & Refactor Roadmap

> 세션 시작 시 `HARNESS_RULES.md` 다음으로 본 파일을 읽어 **직전 완료점 / 다음 액션**을 파악한다.  
> 최종 갱신: **2026-07-16** (KST) · 제품 버전 **v4.3.0**

상태 범례: `[x]` 완료 · `[ ]` 미착수 · `[~]` 진행 중

---

## 완료됨 (2026-07-16 — v4.3.0)

- [x] 프로젝트 강결합 분석 및 하네스 도입 디렉터리 계획 수립
- [x] `.cursorrules` + `HARNESS_RULES.md` 하네스 원칙·세션 프로토콜 명문화
- [x] `compute_fms_snapshot` 공개 API (`analysis_utils.py`)
- [x] FMS pytest 하네스: 순위 / `-999` / NaN / no-network / core 계약
- [x] `tests/fixtures/` 합성 KRW·OHLC·골든 랭크 + `scripts/fixtures/generate_synthetic_panel.py`
- [x] `harness/run_fms_snapshot.py` 오프라인 CLI 러너
- [x] `docs/` 문서 이관 + `TODO.md` + work-plans
- [x] 푸시 전 정리: 디버그 주석 제거, `core`↔`analysis_utils` 경계 분리, 버전 4.3.0 동기화

### 구축한 FMS 하네스 요약 (검증 방법)

| 자산 | 검증 내용 | 실행 |
|------|-----------|------|
| `tests/unit/test_fms_scoring.py` | 골든 순위, CRASHY→-999, OHLC 없음, NaN, yfinance 미호출 | `python -m pytest` |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | (pytest에 포함) |
| `harness/run_fms_snapshot.py` | fixture → FMS 테이블 육안 확인 | `python -m harness.run_fms_snapshot` |
| `scripts/fixtures/generate_synthetic_panel.py` | seed=42 패널 재생성 | 필요 시만 |

---

## 지금 당장 (Next — 우선순위 순)

- [ ] `MarketDataPort` + `YFinanceAdapter` + `FixtureAdapter` (`adapters/market_data.py`)
- [ ] `calculate_fms_for_batch`에서 **다운로드 / 스코어 분리**
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

---

## 빠른 명령

```bash
python -m pytest
python -m harness.run_fms_snapshot
```
