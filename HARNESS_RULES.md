# HARNESS_RULES.md — Harness Engineering 원칙

> **세션 시작 시 최우선 참조 문서.**  
> 이 프로젝트의 모든 코드 수정·기능 추가·버그 수정은 본 문서의 원칙을 따른다.  
> 문서와 코드가 상충하면 우선순위는 **1) 실제 동작 소스코드 → 2) `.cursorrules` → 3) 본 문서 및 `docs/*.md`**.

최종 갱신: 2026-08-08 (KST) · 제품 버전 v5.0.3

---

## 0. 현재 구축된 검증 하네스 (v5.0.x)

### UI / 시장 데이터 캐시 (v5.0.3)

| 자산 | 역할 | 검증 방법 |
|------|------|-----------|
| `adapters/ui_data_bundle.py` (`fingerprint_price_panel` / `DetailViewAtom` / `reconcile_detail_selection`) | 세부보기 심볼↔시계열 원자 결합 · 세션 번들 지문 | `test_ui_panel_fingerprint.py` / `test_detail_view_atom.py` |
| `adapters/price_cache.py` (`needs_refresh` / `DiskPriceCache` / `CachingMarketDataAdapter`) | 배치·UI 공유 디스크 캐시 · last-bar probe | `test_price_cache_freshness.py` |

### FMS / 퀀트 스코어 (오프라인)

| 자산 | 역할 | 검증 방법 |
|------|------|-----------|
| `core/fms_features.py` (`PRODUCTION_ALIVE_PULLBACK_PARAMS` / `score_alive_pullback_from_params` / `score_production_fms_features`) | **v5.0 production SSOT**: alive_pullback 절대 비선형 점수 | `test_fms_alive_pullback_production.py` / `test_nonlinear_mc_features.py` |
| `core/fms_features.py` (`score_legacy_sparse_fms_features` / `cash_like_strength`) | **v4.6–v4.7 archived** sparse+상대Z+현금 게이트 | `test_fms_cash_like_gate.py` / `harness.compare_cash_like_gate` |
| `core/fms.py` (`compute_fms_snapshot` / `momentum_now_and_delta` / `score_fms_from_feature_frame`) | production orchestration · `-999` · ΔFMS | fixture; `analysis_utils` 셔임 |
| `core/fms.py` (`score_legacy_fms_from_feature_frame` / `FmsScoreParams`) | **pre-v4.6 archived formula** (tune 스크립트·회귀만) | `test_fms_params` / `test_fms_vol_tune` / `test_fms_recent_continuation` |
| `core/indicators.py` | `ema` / `returns_pct` / `r_squared_3m` / `ytd_return` / `last_vol_annualized` / `mask_non_positive_prices` / **`harmonize_calendar`(native as-of)** / `align_bday_ffill` | `tests/unit/test_indicators.py` / `test_native_asof_calendar.py` |
| `core/tradeability.py` | True Range 거래적합성 실격 | `tests/unit/test_tradeability.py` |
| `tests/fixtures/synthetic_*.csv` + `golden_fms_ranks.json` | 체크인 Mock 패널 (seed=42) | 골든 순위·실격 |
| `tests/fixtures/cash_like_paths_prices_krw.csv` | 현금성/채권/주식 경로 Mock | 레거시 게이트 + v5 저순위 계약 |
| `tests/unit/test_fms_scoring.py` | 골든 순위 / `-999` / **reference 불변(절대 점수)** / 결측 / yfinance 미호출 | `python -m pytest` |
| `tests/unit/test_fms_alive_pullback_production.py` | 동결 파라미터 · core≡calibration family · `-999` | (pytest 포함) |
| `tests/unit/test_nonlinear_mc_features.py` | SEG_* 비중첩 · MID_DIP · STALE 게이트 · family 등록 | (pytest 포함) |
| `tests/unit/test_fms_recalib_parity.py` | feature-frame scorer ≡ snapshot FMS (동일 패널) | (pytest 포함) |
| `tests/unit/test_fms_features.py` | 3M visible-window 피처 방향성·결측 안전 | (pytest 포함) |
| `tests/unit/test_calibration_session.py` | `saved_at` 기준 최신 완료 세션 선택 (mtime 금지) | (pytest 포함) |
| `tests/unit/test_fms_cash_like_gate.py` | **legacy** sparse+gate · relative-Z 회귀 | (pytest 포함) |
| `tests/unit/test_fms_recent_continuation.py` | **legacy** soft R² / r1_bad continuation (v4.5.1 회귀) | (pytest 포함) |
| `tests/unit/test_fms_params.py` / `test_fms_vol_tune_params.py` / `test_fms_horizon_map.py` | **legacy** params·vol·6M→4M 매핑 | (pytest 포함) |
| `calibration/manifest.py` | 최신 완료 세션·snapshot·ranking hash·development/audit split 고정 | build 시 freshness assert |
| `calibration/ranking_metrics.py` | inversion / Spearman / subset 재서열화 pair-delta | MC/refit 공용 |
| `calibration/nonlinear_formulas.py` | 해석 가능 비선형 family (alive_pullback 포함; core SSOT 위임) | `test_nonlinear_mc_features.py` |
| `calibration/fms_recalib_nonlinear_mc.py` | **기본 원점 경로**: NL 규칙 연계 MC 경쟁 적합 | candidate JSON |
| `calibration/fms_recalib_inspect_patterns.py` | TOP/MID/BOT + natural-language rules JSON | 수동 |
| `calibration/fms_recalib_plot_residuals.py` | 잔차 상위 심볼 3M 차트 PNG | `python -m calibration.fms_recalib_plot_residuals` |
| `calibration/fms_recalib_refit.py` | **레거시** 0점 sparse/GAM/제한 상호작용 | candidate JSON (비기본) |
| `fms_recalib_scratch_candidate.json` (+ scores/residuals/manifest) | 승격 증거 (`promoted_to_production_v5.0.0`) | 회귀 비교용 |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | (pytest 포함) |
| `tests/contract/test_prefilter_not_stricter_than_local.py` | Finviz Perf 사전필터 ≤ 로컬 (배칭용 early cut) | (pytest 포함) |
| `harness/run_fms_snapshot.py` | 동일 fixture 수동 CLI | `python -m harness.run_fms_snapshot` |
| `harness/compare_batch_ui_fms.py` | 배치 vs UI 캘린더 경로 dFMS (동일·연속 실행) | `python -m harness.compare_batch_ui_fms --offline` / `--live` |
| `tests/unit/test_batch_ui_fms_paths.py` | 경로 빌더 bit-identical · coverage 0.5 vs 0.9 드롭 · **native as-of 보존** | (pytest 포함) |
| `tests/unit/test_native_asof_calendar.py` | 다국가 trailing ffill 금지 · 양방향/3시장 clip · FMS 불변 | (pytest 포함) |
| `harness/compare_cash_like_gate.py` | **legacy** 현금성 게이트 기여·영향 비교 | `python -m harness.compare_cash_like_gate` |
| `harness/diagnose_fms_outlier.py` | 단일 티커 FMS 극단치 원인 LIVE 점검 | `python -m harness.diagnose_fms_outlier SYMBOL` |
| `harness/check_relative_ranks.py` | (역사적) 관심종목 상대순위 LIVE 점검 — v5에서는 절대점수 확인용으로만 | `python -m harness.check_relative_ranks` |
| `scripts/fixtures/generate_synthetic_panel.py` | 합성 골든 fixture 재생성기 | 필요 시만 |
| `scripts/fixtures/generate_cash_like_panel.py` | 현금성/채권/주식 경로 fixture 재생성기 | 필요 시만 |
| `scripts/fixtures/prefilter_band_sample_fms.csv` | Finviz 사전필터 경계 밴드 실측 증거 (LIVE 산출) | 수동 참고 |
| `scripts/analyze_prefilter_impact.py` | Finviz 사전필터 tightness 실측 (LIVE; 운영 미import) | `python scripts/analyze_prefilter_impact.py` |

**v5.0.0 검증 요약 (alive_pullback 절대 비선형)**

1. 앱·배치 동일 `momentum_now_and_delta` / `score_production_fms_features` 경로.
2. reference panel 변경해도 FMS 불변 (절대 점수); API는 `reference_prices_krw`를 받지만 무시.
3. 정답셋 full inv 0.146 / Spearman 0.877 vs production-v4.7 benchmark 0.274 / 0.566.
4. 합성 fixture 골든 순위 `TREND_UP > MILD_UP > FLAT > CRASHY(-999)` 유지.
5. calibration `alive_pullback` family score ≡ `core.score_alive_pullback_from_params`.
6. 레거시 sparse+cash gate는 harness에서만 회귀; production 미사용.

**v5.0.3 검증 요약 (2026-08-08 — 세부보기 캐시 + last-bar probe)**

1. DetailViewAtom: symbol == series.name; 없는 심볼 대체 금지; 불일치 fail-closed.
2. 동일 패널 지문에서 UI 세션 번들이 FMS 재계산을 생략.
3. `needs_refresh`: 같은 날짜 HIT · 신규일 MISS · probe None이면 캐시 유지.
4. CachingMarketDataAdapter: 1회 miss 후 디스크 HIT (FixtureAdapter 카운트).
5. 회귀: 전체 pytest; 운영 `cache/` gitignore.

**v5.0.2 검증 요약 (2026-08-07 — 종목별 native as-of)**

1. `harmonize_calendar`는 컬럼별 `last_valid_index` 너머로 ffill하지 않음 (시장 라벨 무관).
2. KR선행·US선행·HK만 앞선 3시장 패널에서 trailing phantom → FMS 불변 계약.
3. `returns_pct` / vol / YTD도 컬럼별 last valid (패널 전역 `iloc[-1]` 재ffill 제거).
4. `app.py` 로컬 harmonize 복제 제거 → `core.indicators` SSOT.
5. 회귀: `test_native_asof_calendar` + 전체 pytest; snapshot 골든 순위 유지.

**v5.0.1 검증 요약 (2026-08-05 — 배치 게이트 + 경로 ΔFMS 하네스)**

1. 배치가 watchlist reference &lt;2 일 때 스캔을 중단하던 v4.7 상대-FMS 게이트를 제거.
   관심종목은 스캔 제외·거래적합성 걸러내기만; 점수에는 reference 미사용.
2. `compare_batch_ui_fms`: 동일 원본에서 UI vs 배치 패널 경로 dFMS —
   LIVE 혼합 24종 max|d|≈0.01–0.014, 순위상관 1.0 → **경로 강제 통일 불필요**.
3. 운영: v5 FMS 추가 잔차/재피팅·사전필터 변경은 pain 명시 전 보류 (`TODO.md` 비범위).
4. 회귀: `test_batch_ui_fms_paths` + 전체 pytest; app/`run_scan_batch` import 스모크.

**v4.7.0 검증 요약 (관심종목 상대 Z — 역사적)**

1. 앱: target=current watchlist, reference=current watchlist; 축별 ungated 기여 평균 0.
2. 배치: target=신규 후보, reference=현재 계좌모드 watchlist.
3. reference 변경 시 FMS 변경; reference 미지정 시 target self-reference.
4. 유효 reference 2개 미만 또는 표준편차 0인 축은 기여 0.
5. 80종 패널 양수 42 / 음수 38; Spearman 0.8917 → 0.8933.

**v4.6.1 검증 요약 (현금성 게이트, 당시 고정 Z 기준)**

1. 합성 현금성 경로 FMS ≈ 2.72 → ≈ −0.58; 주식/장기채/고수익 매끄러운 경로 점수 불변.
2. 승인 80종 캘리브레이션 패널: old/new bit-identical (Spearman/inversion 불변).
3. `cash_strength = low_return × ultra_low_vol × high_smooth`; 양의 품질 보너스만 축소.
4. 계약: `tests/unit/test_fms_cash_like_gate.py` + `harness/compare_cash_like_gate.py`.

### 배치 I/O · 유니버스 (네트워크 없이 단위 검증)

| 자산 | 역할 | 검증 방법 |
|------|------|-----------|
| `tests/unit/test_yf_rate_limit_retry.py` | yfinance `shared._ERRORS` 레이트리밋 감지·재시도 | mock `yf.download` |
| `tests/unit/test_finviz_ticker_normalize.py` | Finviz 티커 첫 글자 중복 보정 | 순수 함수 assert |
| `tests/unit/test_finviz_screener_pagination.py` | Finviz 페이지별 재시도·partial fallback | mock Overview |
| `tests/unit/test_hk_classify.py` | `.HK → HKG` classify | (pytest 포함) |
| `tests/unit/test_hk_fx_conversion.py` | `HKDKRW = USDKRW / HKDUSD` | (pytest 포함) |
| `tests/unit/test_hk_universe_loader.py` | FREE HK 병합 / IRP 제외 | (pytest 포함) |
| `tests/unit/test_market_data_port.py` | `MarketDataPort` 계약: fixture 배치 = 직접 스코어링 FMS 일치, 4-tuple FX, no-network | `FixtureAdapter` 주입 |
| `adapters/market_data.py` | `YFinanceAdapter`(운영) / `FixtureAdapter`(테스트) | `calculate_fms_for_batch(market_data=...)` |
| `scripts/build_hk_universe_from_indices.py` | HK 유니버스 LIVE 재생성 (HSI CSV + HSCEI/HSTECH PDF) | `python scripts/build_hk_universe_from_indices.py` |

검증 명령: `python -m pytest` 및 `python -m harness.run_fms_snapshot`.

### 운영 데이터 커밋 (필수)

아래 파일은 **최신 제품 상태 SSOT**이므로, 로컬에서 변경되었으면 **모든 커밋에 반드시 포함**한다. 코드 전용 커밋이라도 “무관 파일”로 제외하지 않는다.

- `watchlist_free.csv`
- `watchlist_irp.csv`
- `screened_universe.csv`  
운영 코드(`app.py`, `run_scan_batch.py`)는 fixture·테스트 경로를 import하지 않는다.

### 2026-08-02 세션 — 원점 재피팅 절차 전환 (비선형 + MC)

1. **표준 변경**: 자연어 규칙 → 비중첩/고해상도 피처 → 비선형 수식 → 몬테카를로 경쟁.
2. **정답셋**: `cal_fms_20260730_190637` (147종, 3M, inconsistencies=5). 저수익·임의 종목 포함으로 편향 완화.
3. **피처**: `SEG_*` 비중첩 구간 + `PRIOR_SUPPORT_SIGN` (`core/fms_features.py`).
4. **기본 적합 경로**: `fms_recalib_nonlinear_mc.py` (sparse/GAM L-BFGS는 레거시 비교용).
5. **금지**: 자산군/티커 예외 규칙; 승인 전 production 변경.
6. **문서 SSOT**: `docs/FMS_RECALIBRATION_WORKFLOW.md`.

### 2026-07-30 세션 — 현금성 ETF 과대평가 게이트 (v4.6.1)

1. **원인**: v4.6.0 고정 정규화가 저수익·초저변동·고R² 현금성 경로에 무위험형 품질 보너스를 부여.
2. **수정**: `cash_like_strength`로 양의 품질 기여만 억제; `R_3M`·감점·`-999` 불변.
3. **영향**: 80종 캘리브레이션 점수 불변; IRP 스캔 기준 현금성 strength>0.9 약 30종 억제 대상.
4. **회귀**: `test_fms_cash_like_gate.py`; 골든 순위 유지.

### 2026-07-30 세션 — 현재 관심종목 상대 Z-score 복원 (v4.7.0)

1. **의도 복원**: FMS 0을 고정 development 기준이 아니라 현재 계좌 관심종목 기준선으로 정의.
2. **앱**: 관심종목 self-reference; **배치**: 신규 후보를 계좌 watchlist reference와 비교.
3. **안전 처리**: reference 유효값 <2 또는 std≈0이면 해당 축 기여 0; Z ±4 clip 유지.
4. **영향**: 관심종목 구성·계좌모드에 따라 동일 종목 FMS가 달라지는 것이 의도된 동작.
5. **회귀**: reference sensitivity/self fallback/centering/zero variance/batch parity.
6. **푸시 전 정리**: cash fixture 생성기 → `scripts/fixtures/`; `.gitkeep` 인코딩 복구;
   프로덕션 docstring·하네스 README·SSOT 문서 동기화; pytest/import 스모크.

### 2026-07-29 세션 — FMS 원점 재피팅 production 승격 (v4.6.0)

1. **정답셋**: JSON `saved_at` 기준 최신 완료 세션 하나만 사용한다. 과거 세션과 합치지 않는다.
2. **출발점**: production FMS는 benchmark일 뿐이며 후보 점수는 0에서 시작한다.
3. **후보군**: sparse linear / monotone GAM / 제한 상호작용; fold 내부 정규화·L1 선택.
4. **검증**: nested holdout(부분집합 rank를 1…n으로 재서열화), bootstrap, LOO, review label 전 변형.
5. **산출물**: 승인 전 `candidate_only_not_promoted`, 승인 후 `promoted_to_production` 상태를 기록한다.
6. **승격 당시**: 사용자 승인 후 10개 축 sparse-linear 수식과 고정 normalization을 반영했다.
   고정 normalization은 v4.7.0에서 관심종목 상대 Z로 대체되었고 가중치만 유지된다.
7. **상세 절차**: `docs/FMS_RECALIBRATION_WORKFLOW.md`가 재보정 운영 문서 SSOT이다.

### 2026-07-29 세션 — production FMS 최근 우상향 튜닝 (v4.5.1)

1. **원인**: R² below 0.85 + R_1M>30% 시 `r1_bad`가 꾸준한 가속까지 이벤트 급등으로 감점.
2. **수정**: `_r1_conditional_series` — soft R² quality(0.80) + continuation 면제(R_10D and EMA slope > 0); `w_recent`/`w_ema_shape` 소폭 상향.
3. **회귀**: `test_fms_recent_continuation.py` (오프라인 synthetic stale vs recent); 골든 순위 불변.
4. **UI**: 사이드바 FMS 설명을 동일 게이트/면제/단기 축에 맞춤 (`config.FMS_FORMULA` 주석과 동기).
5. **운영/하네스 경계**: `app.py` / `run_scan_batch.py`는 tests·fixture 미import; Mock은 `tests/`·`harness/`만.

### 2026-07-20 세션에서 확정된 FMS 검증 요약

1. **순수 스코어 SSOT** = `core/fms_features.py` (피처·가중치·관심종목 상대 Z) + `core/fms.py` (reference orchestration · `-999` · ΔFMS).
2. **정규화** = 현재 계좌 watchlist median/mean/std + Z ±4 clip. `reference_prices_krw`가 실제 기준 패널이다.
3. **레거시 수식** = `score_legacy_fms_from_feature_frame` / `FmsScoreParams` (tune·회귀 전용; production 경로 아님).
4. **회귀**: 합성 패널 골든 순위 `TREND_UP > MILD_UP > FLAT > CRASHY(-999)`.
5. **사전필터**: Finviz `Quarter Up` / `Half Up` + 로컬 `Perf > 0` (구 Q+10/H+20 폐기).
6. **사전필터 ≤ 로컬 불변식**: Finviz Perf 축 exclusive floor ≤ 로컬 floor (`test_prefilter_not_stricter_than_local`). 사전필터는 배치 시간 절약용 early cut일 뿐, 로컬보다 엄격하면 안 됨.
7. **사전필터 실측 CSV** (`scripts/fixtures/prefilter_band_sample_fms.csv`): Q+10/H+20 시대 스냅샷 — 참고용.
8. **운영 데이터 커밋**: `watchlist_free.csv` / `watchlist_irp.csv` / `screened_universe.csv` 변경 시 모든 커밋에 포함.

### 2026-07-20 세션 — UI (v4.4.8)

- 세부보기 하단 Drawdown: 선택 기간·관심종목 전체 min/max로 y-range 고정 (상단 Rebased 100과 동일; FMS=-999 제외).
- `관심종목 초기화` 버튼 제거 — 운영 실수 방지.

### 가격 / 배당 정책 (확정)

- **수익·FMS:** `auto_adjust=False` + **Adj Close** (배당 조정 총수익)
- **거래적합성 OHLC:** raw High/Low/Close (실제 거래 변동성)
- **비양수 Adj Close:** 스코어링 전 `mask_non_positive_prices`로 NaN 처리 (음수 히스토리→EMA/FMS 폭증 방지; v4.4.9)
- **홍콩 FX (v4.5.0):** `HKG` 종목은 `HKDKRW = USDKRW / HKDUSD`; 수익/FMS·거래적합성 경로는 미국/한국과 동일 정책
- UI 배당 분해 표시는 중기 선택 과제 (`TODO.md`)

### 2026-07-28 세션 — FREE 홍콩 유니버스 + Finviz 복원력 (v4.5.0)

1. **유니버스**: FREE = Finviz US + `korean_universe.csv` + `hongkong_universe.csv` (108종, HSI/HSCEI/HSTECH 합집합).
2. **classify**: `.HK` 접미사 → `HKG`; IRP는 HK 미포함.
3. **FX 경로**: `download_fx()` / `MarketDataPort.get_fx()` → `(USDKRW, USDJPY, JPYKRW, HKDKRW)`.
4. **Finviz hang fix**: `finviz_screener_view_resilient()` — per-page timeout, 5× exponential backoff, `allow_partial` fallback.
5. **회귀 하네스**: `test_hk_*`, `test_finviz_screener_pagination`; 계약 — `update_universe_file` must call resilient helper.
6. **운영/하네스 경계**: `app.py` / `run_scan_batch.py`는 tests·fixture 미import; HK 재생성은 `scripts/` LIVE 전용.


## 1. 목적

KRW Momentum Radar는 FMS(모멘텀 스코어)·거래 적합성 필터·배치 스캔·리캘리브레이션을 다룬다.  
시장 API·UI에 의존하지 않고도 **알고리즘 무결성**을 반복 검증할 수 있어야 한다.  
이를 위해 UI / I/O / 순수 로직을 분리하고, Mock·fixture로 격리 테스트하는 **Harness Engineering**을 표준으로 한다.

---

## 2. 핵심 원칙

### 2.1 관심사 분리 (Decoupling)

| 계층 | 역할 | 허용 | 금지 |
|------|------|------|------|
| `core/` | FMS·지표·필터 등 순수 로직 | pandas/numpy/scipy | yfinance, finviz, requests, streamlit, 네트워크 |
| `adapters/` | 시장 데이터·유니버스·파일 I/O | API 클라이언트, CSV/픽클 | FMS 공식 복제 |
| `app.py` / `run_scan_batch.py` | orchestration만 | core + adapters 조합 | 비즈니스 수식 인라인 복제 |
| `calibration/` | 재보정·튜닝 | production 평가는 core FMS; **0점 scratch**는 비중첩 피처 + 비선형 family + MC (레거시 sparse는 비교용) | 승인 전 production scorer에 후보 섞기; 자산군 예외 규칙 |
| `harness/` | 수동 시나리오 러너 | fixture 주입 | 라이브 API 기본 경로 |
| `tests/` | 자동 검증 | fixture + pytest | 네트워크 필수 단위 테스트 |

과도기에는 `analysis_utils.py` 등이 **re-export 셔임**으로 남을 수 있다.  
물리 이동은 파일 단위로 점진 진행한다 (big-bang 금지).  
Streamlit Cloud 호환을 위해 **`app.py`, `run_scan_batch.py`는 루트에 유지**한다.

### 2.2 UI / 라이브 API 배제 (테스트 경로)

- 단위·회귀 테스트에서 **yfinance / Finviz / 실시간 HTTP를 호출하지 않는다.**
- 스코어링 진입점은 DataFrame 주입형이어야 한다:
  - `compute_fms_snapshot(prices_krw, reference_prices_krw=..., ohlc_data=..., symbols=...)`
  - `momentum_now_and_delta(...)` (위 스냅샷 + ΔFMS)
- `calculate_fms_for_batch`처럼 **다운로드+스코어가 결합된 함수**는 단위 테스트 타깃으로 쓰지 않는다.  
  다운로드와 스코어를 분리한 뒤 스코어만 하네스한다.

### 2.3 Mock 데이터 주입 (Stubs & Adapters)

- 시장 데이터는 **Port / Adapter**로만 주입한다.
  - 프로덕션: `YFinanceAdapter` (예정 / `adapters/`)
  - 테스트: CSV·픽클 `FixtureAdapter` (`tests/fixtures/`, `scripts/fixtures/`)
- 체크인 fixture는 **gitignore 스냅샷에만 의존하지 않는다.**  
  (`fms_calibration_snapshots/`는 로컬 자산일 수 있음)
- Z-score 재현: **production v5.0**은 절대 비선형 점수라 watchlist Z를 쓰지 않는다.
  레거시 sparse 하네스만 watchlist median/mean/std를 사용한다.
  앱은 target self-reference, 배치는 전달된 `reference_prices_krw`를 기준으로 한다.
  동일 fixture에서 target과 reference를 함께 고정해야 점수가 재현된다.

### 2.4 격리된 자동 테스트 하네스

- 위치: `tests/` (pytest), 수동 실험: `harness/`, 보조 스크립트: `scripts/`
- 필수 엣지 케이스 예시:
  - 극단 변동성 / True Range 실격 → FMS = `-999`
  - 결측·전 구간 NaN 컬럼
  - OHLC 없음(필터 스킵) vs OHLC 있음
  - production v4.7: reference 변경 시 점수 변동, 미지정 시 target self-reference
  - reference 유효값 부족/zero variance 축은 기여 0
- `core/` 네트워크 import 금지는 `tests/contract/`로 강제한다.
- **Finviz 사전필터 ≤ 로컬 후처리**: 사전필터는 배치 I/O를 줄이기 위한 early cut이다.
  동일 Perf 축에서 Finviz exclusive floor가 로컬보다 높으면(더 엄격하면) 안 된다.
  SSOT 상수·헬퍼: `universe_utils` (`FINVIZ_PERF_*_LABEL`, `LOCAL_PERF_*_GT`,
  `assert_prefilter_not_stricter_than_local`). 계약:
  `tests/contract/test_prefilter_not_stricter_than_local.py`.
- **TDD 순서:** (1) 순수 I/O 인터페이스 정의 → (2) fixture·테스트 작성 → (3) 구현 → (4) 엔트리포인트 연결

### 2.5 FMS 단일 소스 오브 트루스

- Production 공식의 유일한 구현: `core.fms.compute_fms_snapshot` / 내부 `_mom_snapshot`  
  (`analysis_utils`는 re-export 셔임; 지표는 `core/indicators.py`, 필터는 `core/tradeability.py`)
- Production baseline·legacy incremental tuning은 **동일 core API**만 사용한다.
- **원점 scratch 후보**는 공유 순수 피처(비중첩 `SEG_*` 포함) 위에서
  **비선형 family + 몬테카를로**로 0점 출발 적합한다. production은 benchmark일 뿐이다.
  사용자 승인 전에는 `core/fms.py`에 반영하지 않는다.
  레거시 `fms_recalib_refit.py`(sparse/GAM)는 비교·회귀용이다.
- `fms_recalib_evaluate_formulas.f_current` / 과거 tune 스크립트 내 독립 `fms_score`처럼  
  **production 공식 복제는 금지**한다. 발견 시 통합·삭제 대상으로 기록한다 (`TODO.md`).
  (v4.4.6: weights tune `fms_score`는 `score_fms_from_feature_frame(..., params=...)` 위임으로 통합됨.  
  `fms_recalib_tune_vol_penalty.fms_score_with_vol_params`는 단순화 탐색용으로 잔류 — 승자는 core `vol_*` params로 승격.)

---

## 3. FMS 산출 검증 룰

1. **입력은 항상 주입 가능해야 한다.** 라이브 다운로드는 adapter 레이어에서만.
2. **골든 테스트:** `tests/fixtures/golden_fms_ranks.json` 등 고정 기댓값과 순위·실격을 비교한다.
3. **공식 변경 시:**
   - 먼저 하네스에서 전/후 점수·순위·inversion 지표를 비교한다.
   - 의도된 변경이면 골든 fixture를 **명시적으로** 갱신하고 `CHANGELOG.md` / `TODO.md`에 사유를 남긴다.
   - “테스트만 맞추기” 위한 silent golden 변경 금지.
4. **필터(`-999`)와 스코어를 혼동하지 않는다.** OHLC fixture 유무를 테스트 이름·문서에 명시한다.
5. **재현성:** RNG·시드·날짜 인덱스·컬럼 순서를 fixture에 고정한다.

---

## 4. 백테스트 / 리캘리브레이션 검증 룰

1. 피처 테이블·세션 JSON·스냅샷은 **읽기 전용 입력**으로 취급한다.
   production benchmark는 core API로, scratch 후보는
   `fms_recalib_nonlinear_mc.py`(기본) 또는 레거시 `fms_recalib_refit.py`로 계산한다.
2. 워크플로 문서: [`docs/FMS_RECALIBRATION_WORKFLOW.md`](docs/FMS_RECALIBRATION_WORKFLOW.md)
   — 자연어 → 비중첩 피처 → 비선형 → MC 경쟁이 표준이다.
3. 튜닝 결과가 라이브 FMS와 어긋나면 **튜닝 쪽 복제 공식을 폐기**하고 production 경로에 맞춘다.
4. 오프라인 실험 스크립트는 `scripts/` 또는 `harness/` / `calibration/`에 두고,
   프로덕션 엔트리포인트에 실험 코드를 남기지 않는다.
5. 사람 랭킹과 알고리즘 점수를 비교할 때도 **동일 reference 패널**을 사용한다.
6. 재보정 정답셋은 **최종 저장 시각이 가장 최신인 완료 세션 하나만** 사용한다.
7. **몬테카를로는 비선형 후보의 주 optimizer**다. 자산군 예외 규칙은 금지한다.

---

## 5. 세션 운영 프로토콜

새 Cursor 세션을 열면 에이전트는 다음을 수행한다:

1. **본 문서(`HARNESS_RULES.md`) 숙지**
2. **컨텍스트 동기화:** 소스코드 → `.cursorrules` → `docs/*.md` 우선순위로 진실 파악
3. **하네스 자산 파악:** `tests/`, `harness/`, `scripts/`, fixtures 현황과 보호/취약 영역 식별
4. **히스토리:** `CHANGELOG.md`, `TODO.md`, `docs/work-plans/`로 직전 완료점·다음 과제 확인
5. **시간 동기화:** 시스템 날짜·시각(KST) 확인 후 문서·로그·백업에 사용
6. **브리핑:** “준비됐습니다” 대신 아래 3항목을 짧게 보고
   - 주요 하네스/테스트 자산 현황
   - 직전 완료 지점과 현재 병목
   - 오늘의 첫 액션 플랜 (코드 전에 어떤 검증부터 실행할지)

---

## 6. 디렉터리 역할 요약

```
HARNESS_RULES.md     # 본 원칙 (세션 시작 1순위)
TODO.md              # 진행 체크리스트
docs/                # 설계·워크플로·배포·작업 일지
tests/               # pytest 자동 하네스 + fixtures
harness/             # 수동 CLI 시나리오 러너
scripts/             # 일회성/보조 스크립트·fixture 생성기
core/                # 순수 로직 (목표)
adapters/            # 외부 I/O (목표)
calibration/         # 재보정 패키지 (목표)
```

상세 작업 상태는 [`TODO.md`](TODO.md), 문서 인덱스는 [`docs/README.md`](docs/README.md)를 본다.
