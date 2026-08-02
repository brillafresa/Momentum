# TODO — Harness Engineering & Refactor Roadmap

> 세션 시작 시 `HARNESS_RULES.md` 다음으로 본 파일을 읽어 **직전 완료점 / 다음 액션**을 파악한다.  
> 최종 갱신: **2026-08-02** (KST) · 제품 버전 **v5.0.0**

상태 범례: `[x]` 완료 · `[ ]` 미착수 · `[~]` 진행 중

---

## 완료됨

### 2026-08-02 — v5.0.0 (alive_pullback 원점 재피팅 승격)

- [x] 정답셋 cal_fms_20260730_190637 → NL·비중첩 SEG_*·비선형 MC → `alive_pullback` 선정
- [x] `STALE_AFTER_RUN` 최근-약세 게이트 + residual features / alive_pullback family
- [x] production SSOT 승격 (`core/fms_features.py`); 레거시 sparse+cash는 harness 보존
- [x] residual plot 스키마 호환; pytest unit/contract; app/batch import 스모크
- [x] CHANGELOG / HARNESS_RULES / TODO / .cursorrules / docs 동기화

### 2026-07-30 — v4.7.0 (현재 관심종목 상대 Z-score 복원)

- [x] 10축 normalization을 고정 development 통계 → 현재 계좌 관심종목 median/mean/std로 변경
- [x] 앱 self-reference / 배치 account-watchlist reference 경로 연결
- [x] reference 변경·자기 centering·zero variance·batch parity 회귀
- [x] 현금성 게이트·`-999`·골든 순위 유지; 80종 상대점수 영향 검증
- [x] 푸시 전 정리: fixture 생성기 `scripts/fixtures/` 이관, 데드 docstring 정리,
  하네스/문서 SSOT 동기화, pytest·import 스모크

### 2026-07-30 — v4.6.1 (현금성 ETF FMS 과대평가 게이트)

- [x] 저수익∧초저변동∧고R² 현금성 강도 게이트 (양의 품질 보너스만 억제)
- [x] fixture·계약: `test_fms_cash_like_gate.py` / `cash_like_paths_prices_krw.csv`
- [x] 영향: 80종 bit-identical; 합성 현금성 ≈2.72→≈−0.58
- [x] `harness/compare_cash_like_gate.py`; config/UI/HARNESS/CHANGELOG 동기화

### 2026-07-29 — v4.6.0 (FMS 원점 재피팅 production 승격)

- [x] 최신 완료 세션 하나를 `saved_at` 기준으로 고정하고 manifest/hash 생성
- [x] 기존 FMS를 benchmark로만 두고 0점 출발 sparse/GAM/제한 상호작용 후보 피팅
- [x] nested holdout·bootstrap·LOO·review label 전 변형 검증
- [x] development 잔차 차트 검토 및 일반화 피처 추가 검토
- [x] `fms_recalib_scratch_candidate.json` 후보 보고 생성
- [x] 사용자 후보 검토·승인 (`promoted_to_production`)
- [x] 10개 축 sparse-linear 공식·고정 정규화·±4 clip을 production SSOT에 승격
- [x] 앱·배치·feature-frame parity, 거래적합성 `-999`, UI 도움말 동기화
- [x] 푸시 전 정리: dead code(`_PEER_W_*`, unused zscore helpers, app 중복 지표), harness 경계,
  승격 증거 tracked / legacy incremental JSON gitignore, docs SSOT, pytest/import 스모크

---

### 2026-07-29 — v4.5.1 (FMS 최근 우상향 가중 튜닝)

- [x] `r1_bad` 면제: R_10D>0 ∧ EMA20 slope>0 이면 이벤트 급등 감점 제외
- [x] R² quality soft gate (center 0.80 smoothstep); `_r1_conditional_series` SSOT
- [x] `w_recent` +25%, `w_ema_shape` +15%
- [x] 회귀: `test_fms_recent_continuation.py`; 골든 순위 유지
- [x] UI 좌측 [도구 및 도움말] FMS 수식 설명 동기화
- [x] 푸시 전 정리: 프로덕션↔하네스 경계·문서 SSOT·pytest/import 스모크

### 2026-07-28 — v4.5.0 (FREE 홍콩 유니버스 + Finviz 페이지네이션 복원력)

- [x] `hongkong_universe.csv` — HSI/HSCEI/HSTECH 합집합 108종 (수동 관리)
- [x] FREE 유니버스 병합: screened + korean + hongkong; IRP 불변
- [x] `.HK → HKG` classify; HKD→KRW FX (`HKDKRW = USDKRW / HKDUSD`)
- [x] `finviz_screener_view_resilient()` — 마지막 페이지 hang/fix; 계약 테스트
- [x] 회귀: `test_hk_*`, `test_finviz_screener_pagination`, `test_market_data_port` 4-tuple FX
- [x] 유지보수: `scripts/build_hk_universe_from_indices.py` (LIVE 재생성; 운영 미import)
- [x] 배치: `--skip-universe-update` (개발용 Finviz 갱신 생략)

### 2026-07-27 — v4.4.9 (음수 Adj Close FMS 폭증 수정)

- [x] `mask_non_positive_prices` + `_mom_snapshot` / public FMS API 진입점 적용
- [x] 회귀: `test_negative_adj_close_history_does_not_inflate_fms` (381560.KS류)
- [x] 진단: `harness/diagnose_fms_outlier.py` (LIVE 수동 점검)

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
| `tests/unit/test_fms_cash_like_gate.py` | 현금성 게이트 · relative-Z centering · zero variance | (pytest 포함) |
| `tests/unit/test_fms_scoring.py` | 골든 순위, CRASHY→-999, **reference 상대평가**, NaN, yfinance 미호출 | `python -m pytest` |
| `tests/unit/test_fms_recent_continuation.py` | soft R² quality / r1_bad continuation 면제 / stale vs recent | (pytest 포함) |
| `tests/contract/test_prefilter_not_stricter_than_local.py` | Finviz Perf 사전필터 ≤ 로컬 | (pytest 포함) |
| `tests/unit/test_fms_horizon_map.py` | 복리/√t 매핑, R²=63d 불변 | (pytest 포함) |
| `tests/unit/test_fms_params.py` | params 기본값=production, 오버라이드, tune 위임 | (pytest 포함) |
| `tests/unit/test_fms_recalib_parity.py` | recalib feature→score = production snapshot FMS | (pytest 포함) |
| `tests/unit/test_tradeability.py` | True Range 실격·엣지 + analysis_utils 셔임 | (pytest 포함) |
| `tests/unit/test_indicators.py` | `core.indicators` EMA/returns/R² + analysis_utils 셔임 | (pytest 포함) |
| `tests/unit/test_yf_rate_limit_retry.py` | 429/`shared._ERRORS` 재시도 | (pytest 포함) |
| `tests/unit/test_finviz_ticker_normalize.py` | Finviz 티커 첫글자 중복 보정 | (pytest 포함) |
| `tests/unit/test_market_data_port.py` | FixtureAdapter 배치 = 직접 스코어링, 4-tuple FX, no-network | (pytest 포함) |
| `tests/unit/test_hk_classify.py` | `.HK → HKG` classify | (pytest 포함) |
| `tests/unit/test_hk_fx_conversion.py` | HKD→KRW FX 경로 | (pytest 포함) |
| `tests/unit/test_hk_universe_loader.py` | FREE HK 병합 / IRP 제외 | (pytest 포함) |
| `tests/unit/test_finviz_screener_pagination.py` | Finviz 페이지 재시도·partial fallback | (pytest 포함) |
| `scripts/build_hk_universe_from_indices.py` | HK 유니버스 LIVE 재생성 (운영 미import) | `python scripts/build_hk_universe_from_indices.py` |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | (pytest 포함) |
| `harness/run_fms_snapshot.py` | fixture → FMS 테이블 육안 확인 | `python -m harness.run_fms_snapshot` |
| `harness/compare_cash_like_gate.py` | 현금성 게이트 old/new 영향 | `python -m harness.compare_cash_like_gate` |

상세 SSOT: [`HARNESS_RULES.md`](HARNESS_RULES.md) §0.

---

## 지금 당장 (Next — 우선순위 순)

- [x] **2026-08-02 원점 재피팅 승격**: `alive_pullback` → production v5.0.0 (실사용 피드백 후 추가 라운드 검토)
- [x] LIVE IRP/FREE 배치로 v4.7.0 상대 Z + 현금성 게이트 재랭킹 확인 (사용자 직접 실행)
- [x] (선택) get_filter_debug_info → core/tradeability.py 동반 이전
- [x] (선택) ms_recalib_tune_vol_penalty 단순화 본문 정리
- [x] ms_recalib_*.py → calibration/ 점진 이동

---

## 중기 (공식 드리프트 제거)

- [x] `calibration_utils.py` → `calibration/session.py` + 셔임
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
python -m harness.compare_cash_like_gate
python fms_recalib_build_features.py
python -m calibration.fms_recalib_inspect_patterns
python fms_recalib_nonlinear_mc.py
```
