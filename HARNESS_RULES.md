# HARNESS_RULES.md — Harness Engineering 원칙

> **세션 시작 시 최우선 참조 문서.**  
> 이 프로젝트의 모든 코드 수정·기능 추가·버그 수정은 본 문서의 원칙을 따른다.  
> 문서와 코드가 상충하면 우선순위는 **1) 실제 동작 소스코드 → 2) `.cursorrules` → 3) 본 문서 및 `docs/*.md`**.

최종 갱신: 2026-07-28 (KST) · 제품 버전 v4.5.0

---

## 0. 현재 구축된 검증 하네스 (v4.5.0)

### FMS / 퀀트 스코어 (오프라인)

| 자산 | 역할 | 검증 방법 |
|------|------|-----------|
| `core/fms.py` (`compute_fms_snapshot` / `momentum_now_and_delta` / `score_fms_from_feature_frame` / `FmsScoreParams`) | production 스코어 + recalib/tune feature→score | fixture; `analysis_utils` 셔임 |
| `core/indicators.py` | `ema` / `returns_pct` / `r_squared_3m` / `ytd_return` / `last_vol_annualized` / `mask_non_positive_prices` | `tests/unit/test_indicators.py` |
| `core/tradeability.py` | True Range 거래적합성 실격 | `tests/unit/test_tradeability.py` |
| `tests/fixtures/synthetic_*.csv` + `golden_fms_ranks.json` | 체크인 Mock 패널 (seed=42) | 골든 순위·실격 |
| `tests/unit/test_fms_scoring.py` | 순위 / `-999` / 결측 / yfinance 미호출 / 셔임 | `python -m pytest` |
| `tests/unit/test_fms_recalib_parity.py` | recalib `f_current` = production FMS | (pytest 포함) |
| `tests/unit/test_fms_params.py` | params 기본값=production / 오버라이드 / tune 위임 | (pytest 포함) |
| `tests/unit/test_fms_horizon_map.py` | 6M→4M 복리/√t 매핑 · R²=63d 불변 | (pytest 포함) |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | (pytest 포함) |
| `tests/contract/test_prefilter_not_stricter_than_local.py` | Finviz Perf 사전필터 ≤ 로컬 (배칭용 early cut) | (pytest 포함) |
| `harness/run_fms_snapshot.py` | 동일 fixture 수동 CLI | `python -m harness.run_fms_snapshot` |
| `harness/diagnose_fms_outlier.py` | 단일 티커 FMS 극단치 원인 LIVE 점검 | `python -m harness.diagnose_fms_outlier SYMBOL` |
| `scripts/fixtures/generate_synthetic_panel.py` | fixture 재생성기 | 필요 시만 |
| `scripts/fixtures/prefilter_band_sample_fms.csv` | Finviz 사전필터 경계 밴드 실측 증거 (LIVE 산출) | 수동 참고 |
| `scripts/analyze_prefilter_impact.py` | Finviz 사전필터 tightness 실측 (LIVE; 운영 미import) | `python scripts/analyze_prefilter_impact.py` |

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

### 2026-07-20 세션에서 확정된 FMS 검증 요약

1. **순수 스코어 SSOT** = `core/fms.py` (가격 패널 → `compute_fms_snapshot`; 피처 테이블 → `score_fms_from_feature_frame`).
2. **장기 축** = `R_4M`(84d); 게이트/quality는 복리 매핑, `gate_r4_w`는 √t; `w_r4`는 Z 불변으로 유지.
3. **파라미터 SSOT** = `FmsScoreParams` / `production_fms_score_params()`; tune은 `params=` 주입만.
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
| `calibration/` | 재보정·튜닝 | **core FMS 단일 API** 호출 | `f_current` 등 독립 공식 포크 |
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
- Z-score 재현을 위해 **타깃 패널과 `reference_prices_krw`를 함께 고정**한다.

### 2.4 격리된 자동 테스트 하네스

- 위치: `tests/` (pytest), 수동 실험: `harness/`, 보조 스크립트: `scripts/`
- 필수 엣지 케이스 예시:
  - 극단 변동성 / True Range 실격 → FMS = `-999`
  - 결측·전 구간 NaN 컬럼
  - OHLC 없음(필터 스킵) vs OHLC 있음
  - 참조 분포 변경 시 점수 변동
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
- 리캘리브레이션·튜닝·백테스트는 **동일 API** 또는 공유 feature→score 경로만 사용한다.
- `fms_recalib_evaluate_formulas.f_current` / 과거 tune 스크립트 내 독립 `fms_score`처럼  
  **공식 복제는 금지**한다. 발견 시 통합·삭제 대상으로 기록한다 (`TODO.md`).  
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

1. 피처 테이블·세션 JSON·스냅샷은 **읽기 전용 입력**으로 취급하고, 스코어는 core API로 계산한다.
2. 워크플로 문서: [`docs/FMS_RECALIBRATION_WORKFLOW.md`](docs/FMS_RECALIBRATION_WORKFLOW.md)
3. 튜닝 결과가 라이브 FMS와 어긋나면 **튜닝 쪽 복제 공식을 폐기**하고 production 경로에 맞춘다.
4. 오프라인 실험 스크립트는 `scripts/` 또는 `harness/`에 두고, 프로덕션 엔트리포인트에 실험 코드를 남기지 않는다.
5. 사람 랭킹(mergesort 세션)과 알고리즘 점수를 비교할 때도 **동일 reference 패널**을 사용한다.

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
