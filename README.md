# KRW Momentum Radar

⚡ **KRW Momentum Radar v5.0.5**는 다국가 주식 시장의 모멘텀을 실시간으로 분석하고 시각화하는 Streamlit 웹 애플리케이션입니다.

## 🌟 주요 기능

### 📊 가속 보드

- FMS(Fast Momentum Score) 기반 실시간 모멘텀 분석
- **3M alive_pullback 전략**: 비중첩 구간 수익·이전 추세 지지·절대수익 바닥·정체/급등 패널티를 결합한 비선형 FMS
- 1일/5일 가속도 변화량 추적
- 상위 N개 종목의 모멘텀 랭킹

### 📈 비교 차트

- 다국가 종목들의 KRW 환산 가격 비교
- 로그 스케일로 넓은 범위의 수익률 비교
- 기간별 성과 비교 (1M, 3M, 6M, 1Y, 2Y)

### 🎯 수익률-변동성 이동맵

- **정적 모드**: 1개월 전 → 어제 → 오늘의 이동 경로 시각화
- **애니메이션 모드**: 최근 10일/20일의 실시간 이동 추적 (종목명 표시)
- **로그 스케일**: 세로축(수익성) 로그 스케일로 넓은 범위의 수익률 비교 (항상 적용)
- 꼬리 효과로 과거 경로 추적 가능 (사이드바에서 설정)

### 📋 상세 분석

- 개별 종목의 EMA(20, 50, 200) 분석 및 Rebased(100) 기준 5거래일 이동평균선(보조)
- 최대 낙폭(Drawdown) 추적
- 모멘텀 상태 배지 시스템
- **종목 네비게이션**: 정렬된 순서로 이전/다음 종목 이동 및 맨 끝으로 이동
- **차트 기간 연동**: 선택된 차트 기간에 맞춰 차트 표시

### 📁 관심종목 관리

- **계좌 모드별 관리**: 자유투자계좌와 퇴직연금IRP 모드별로 독립적인 관심종목 관리
- **파일 업로드/다운로드**: CSV 파일로 관심종목 백업 및 복원
- **수동 관리**: 개별 종목 추가/삭제 (즉시 파일 저장)
- **FMS 기반 재평가**: 저성과 종목 자동 식별 및 제거 제안
- **실시간 동기화**: 파일 변경 시 자동 새로고침

### 🚀 신규 종목 탐색

- **유니버스 파일 관리**: 스크린된 유니버스 파일 업로드/다운로드
- **실시간 진행률 표시**: 유니버스 업데이트 및 FMS 스캔 과정의 실시간 진행률 표시
- **FMS 스캔**: 사전 필터링된 목록에서 모멘텀 상위 종목 탐색
- **스캔 결과 영구 저장**: FMS ≥ 0.0(절대 점수 저장 하한) 종목을 자동으로 파일에 저장하여 세션 간 유지
- **동적 후보 리스트**: 종목 추가 시 후순위 종목이 자동으로 후보로 올라오는 스마트 관리
- **페이징 시스템**: 5~30개 선택 가능한 페이지당 표시 종목 수로 대량 후보 효율적 탐색
- **FMS 임계값 필터링**: 슬라이더로 원하는 FMS 점수 이상의 종목만 표시
- **저장된 결과 로드**: 이전 스캔 결과를 언제든지 불러와서 계속 탐색 가능
- **원클릭 추가**: 발견된 종목을 관심종목에 즉시 추가 (UI 리로드 없이)
- **[신규] 배치 스캔 결과 UI**: 사이드바의 "📦 배치 스캔 관리"에서 배치 결과를 FMS 순으로 확인하고 원클릭으로 관심종목 추가

### 🚀 진정한 전체 시장 탐색 엔진 (NEW!)

- **실시간 유니버스 스크리닝**: Finviz.com 기반으로 미국 전체 시장(8,000+ 종목)에서 유망주 발굴
- **2단계 추진 로켓 방식**: 사전 필터링 + 온디맨드 FMS 스캔으로 성능 최적화
- **동적 스크리닝**: 가격 $5+, 거래량 200K+, 1개월 수익률 0%+ 등 다중 필터 자동 적용
- **레버리지/인버스 ETF 제외**: 모멘텀의 본질을 흐리는 레버리지형 ETF 자동 제외
- **FMS 기반 랭킹**: 강력한 모멘텀을 가진 신규 종목 자동 발견
- **원클릭 추가**: 발견된 종목을 관심종목에 즉시 추가

### 📋 동적 관심종목 관리 (NEW!)

- **영구 저장**: 관심종목이 세션 종료 후에도 유지
- **실시간 관리**: 직관적인 UI로 관심종목 추가/삭제
- **자동 편출 제안**: 저성과 종목 자동 식별 및 제거 제안

## 🚀 빠른 시작

### 로컬 실행

Windows (권장)

```bat
:: 저장소 클론 후 프로젝트 폴더에서
start.bat
```

일반(수동) 방법

```bash
# 저장소 클론
git clone <repository-url>
cd Momentum

# 가상환경 생성/활성화 및 의존성 설치 (Python 3.11 권장)
py -3.11 -m venv venv || python -m venv venv
./venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 애플리케이션 실행 (포트 8501 사용 권장)
streamlit run app.py --server.port 8501
```

### Streamlit Cloud 배포

1. GitHub에 저장소 푸시
2. [Streamlit Cloud](https://share.streamlit.io)에서 새 앱 생성
3. 저장소 연결 및 `app.py` 파일 지정
4. 자동 배포 완료!

## 📦 포함된 시장

### 🇺🇸 미국 (USD)

- 주요 ETF: SPY, QQQ, VOO, DIA
- 섹터 ETF: XLK, XLF, XLV, SOXX
- 테마 ETF: ICLN, BOTZ, IBB
- 개별주: NVDA, GOOGL

### 🇰🇷 한국 (KRW)

- 대형주: 삼성전자(005930.KS)
- 국내 지수 ETF: KODEX, TIGER, ARIRANG 1배 지수 및 인버스 상품

### 🇯🇵 일본 (JPY)

- 개별주: 2563.T

## 🔧 설정 옵션

### 차트 기간

- 1M, 3M, 6M, 1Y, 2Y 선택 가능

### 정렬 기준

- **ΔFMS(1D)**: 1일 가속도 변화
- **ΔFMS(5D)**: 5일 가속도 변화
- **FMS(현재)**: 현재 모멘텀 점수
- **1M 수익률**: 1개월 수익률

### 표시 옵션

- Top N: 5~60개 종목 선택
- 꼬리 길이: 0~10일 이동 경로 (사이드바에서 설정)
- 수익률/변동성 창: 21, 42, 63일 선택 가능 (사이드바에서 설정)

## 🧰 배치 스캔 사용 (요약)

- 배치 스캔은 정확도 중심의 오프라인 계산입니다. 실행/상태 확인은 앱 사이드바의 "📦 배치 스캔 관리"에서 가능합니다.
- Windows 작업 스케줄러 설정 방법은 [`docs/README_BATCH.md`](docs/README_BATCH.md)를 참조하세요.
- 앱/배치 모두 동일한 단일 FMS/필터 로직(`core/fms.py` · `core/fms_features.py`)을 사용합니다.
- yfinance 레이트리밋 발생 시 지수 백오프로 최대 10회 재시도하며, 상장폐지/데이터 없음은 건너뜁니다.
- **다국가 스캔**: USA(Finviz 스크리닝) + Korea(KOSPI 200 + KOSDAQ 150 + 국내 지수 ETF 1배/인버스) 통합 스캔 지원
- **계좌 모드별 스캔**: 자유투자계좌(FREE)와 퇴직연금IRP(IRP) 모드별로 다른 유니버스 스캔
  - FREE 모드: 미국(Finviz) + 한국(KOSPI200/KOSDAQ150/국내상장 ETF 전 종목)
  - IRP 모드: 국내상장 ETF 전 종목 (korean_etf_univers.csv)
  - 작업 스케줄러에서 파라미터 없이 실행 시 두 모드 모두 자동 실행

## 📊 지표 설명

### FMS (Fast Momentum Score)

#### 최신 3개월 경로 기반 alive_pullback 전략

v5.0.0 FMS는 정답셋 `cal_fms_20260730_190637`(n=147)에서
NL→비중첩 피처→비선형 MC로 피팅한 **`alive_pullback`** 절대 점수입니다.
관심종목 상대 Z-score는 사용하지 않습니다.

- 가산: 최근 1주 수익·회복(alive), 중간 조정 후 회복(`MID_DIP_RECOVERY`),
  이전 추세 지지, 절대 `R_3M` softplus 바닥, 최근 상승일 폭·grind
- 감점: 최근이 약할 때의 `STALE_AFTER_RUN`, 단발 급등 비중(`RECENT_JUMP_SHARE_5D`)
- 앱·배치 동일 절대 수식; 거래적합성 필터의 `FMS=-999` 정책은 기존과 동일
- SSOT: `core/fms_features.py` (`PRODUCTION_ALIVE_PULLBACK_PARAMS`)

## 🔁 FMS 재보정(원점 재피팅)

재보정은 **최종 `saved_at`이 가장 최신인 완료 세션 하나**만 사용합니다.
과거 정답셋을 합치지 않으며, production FMS는 benchmark로만 사용합니다.
새 후보 점수는 0에서 시작합니다.

### 1. UI에서 정답셋 수집

1. 앱의 **FMS 재보정** 섹션에서 새 세션을 시작해 가격 snapshot을 고정합니다.
2. Rebased 100 로그 차트와 Drawdown을 비교해 merge sort를 완료합니다.
3. 인접 순위 약 10%를 재검토하고 세션이 `phase == "done"`이 될 때까지 저장합니다.

### 2. 최신 세션 manifest와 피처 생성

```bash
python fms_recalib_build_features.py
```

- JSON `saved_at` 기준 최신 완료 세션 선택
- ranking/snapshot hash와 development/audit split 기록
- 사용자가 본 3M window에서 해석 가능한 피처 생성

### 3. 원점 후보 피팅

```bash
python fms_recalib_refit.py
```

- sparse linear, monotone GAM, 제한 상호작용 모델 비교
- fold 내부 정규화·L1 선택과 pairwise ordering loss
- one-standard-error 규칙으로 단순 모델 우선
- nested holdout, symbol bootstrap, LOO, 모든 review label 변형 검증

개발 잔차는 다음 명령으로 고정 snapshot 차트를 확인합니다.

```bash
python -m calibration.fms_recalib_plot_residuals
```

### 4. 후보와 production의 분리

- 후보 산출물: `fms_recalib_scratch_candidate.json`
- 2026-07-29 승인 후보는 v4.6.0 production으로 승격되었습니다.
- 다음 재보정에서도 `status=candidate_only_not_promoted`인 동안에는 당시 production을 계속 사용합니다.
- inversion↓, Spearman↑, pair-delta↓와 안정성 검증을 모두 보고한 뒤
  사용자 컨펌을 받아야만 `core/fms.py`에 승격합니다.

전체 절차와 현재 후보 수식·검증 한계는
[`docs/FMS_RECALIBRATION_WORKFLOW.md`](docs/FMS_RECALIBRATION_WORKFLOW.md)를 참고하십시오.

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Data**: yfinance (Yahoo Finance API), finvizfinance (Finviz 스크리닝)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Timezone**: pytz (KST 기준)

## 📝 버전 히스토리

### v5.0.5 (현재)

- 배치 1y 캐시를 UI 2y 요청에 HIT하던 period 불일치 수정 (ITGR 등 「데이터 부족」 탈락)
- 검증: `cache_covers_request` pytest · UI ITGR 보드/차트

### v5.0.4

- 캐시 adapter cold-path probe 제거 → 배치 시 미국 청크 레이트 실패로 USA가 신규탐색에서 사라지던 문제 수정
- 검증: `test_price_cache_freshness` · `harness.smoke_multi_market_batch` · `harness.smoke_usa_first_batch` · FREE 풀 배치/UI

### v5.0.3

- 세부보기 DetailViewAtom(심볼↔차트 원자 결합) + 세션 FMS 메모
- 배치·UI 공유 디스크 캐시 + last-bar probe 신선도

### v5.0.2


- **수정**: 다국가 패널 `harmonize_calendar`가 종목별 마지막 실거래일 너머로 ffill하던
  trailing phantom 봉을 제거 (컬럼별 native as-of). KR/US/HK 어느 쪽이 앞서도 SEG_* 창이
  밀리지 않음 (ITGR: UI 10.24 → 배치와 동일 ~5.15).
- `returns_pct` / `last_vol_annualized` / `ytd_return`도 컬럼별 last valid 기준으로 통일.
- 회귀: `tests/unit/test_native_asof_calendar.py`

### v5.0.1

- **배치 v5 절대 FMS 문구/게이트**: watchlist≥2 “relative-FMS” 중단 제거 (관심종목 비어도 스캔 계속)
- **배치↔UI 경로 ΔFMS 하네스**: `harness/compare_batch_ui_fms.py` + unit 계약 (연속 실행 시 max|d|≈0.01·순위 불변 → 경로 통일 보류)
- **문서**: README_BATCH / TODO / work-plans / HARNESS_RULES 동기화; v5 ops 잔차 라운드 보류

### v5.0.0

- **alive_pullback 원점 재피팅 승격**: NL→SEG_*→비선형 MC 후보를 production SSOT로 적용
- 절대 경로 점수(관심종목 상대 Z / 현금성 게이트 제거); `-999` 거래적합성 유지
- 하네스: `test_fms_alive_pullback_production`, `test_nonlinear_mc_features`, residual plot 호환

### v4.7.0

- **관심종목 상대 Z-score 복원**: 각 축을 현재 계좌모드 관심종목의 평균·표준편차로 정규화
- 앱은 현재 관심종목끼리, 배치는 신규 후보를 현재 관심종목 기준으로 평가
- `FMS=0`은 고정 development 기준이 아니라 현재 관심종목의 상대 기준선
- reference 변경·자기 참조 centering·배치 port parity 회귀 하네스 추가

### v4.6.1

- **현금성 경로 게이트**: 저수익 ∧ 초저변동 ∧ 고R²일 때 품질 축의 양의 보너스만 억제 (`R_3M`·감점 불변)
- KOFR/CD금리/머니마켓 ETF가 v4.6.0에서 상위권을 독점하던 분포 외삽 수정
- 승인 80종 캘리브레이션 패널 점수는 bit-identical 유지
- 하네스: `test_fms_cash_like_gate.py`, `harness/compare_cash_like_gate.py`

### v4.6.0

- 최신 80종 정답셋의 zero-based sparse-linear FMS를 production으로 승격
- 10개 3M-window 축, 고정 normalization, ±4 clip을 `core/fms_features.py` SSOT로 적용
- 앱·배치·feature-frame scorer parity와 거래적합성 `-999` 유지
- 좌측 도움말과 재보정 문서를 실제 production 수식에 맞게 갱신

### v4.5.1

- **FMS 최근 우상향 튜닝**: 연속 상승 시 `r1_bad` 면제, R² quality soft gate(0.80), `w_recent`/`w_ema_shape` 소폭 상향
- **UI**: 좌측 [도구 및 도움말] FMS 설명을 동일 게이트/단기 연속 축에 맞춤
- **하네스**: `test_fms_recent_continuation.py`

### v4.5.0

- **FREE 모드 홍콩 유니버스**: `hongkong_universe.csv` (HSI·HSCEI·HSTECH, 108종) + HKD→KRW FX 경로
- **Finviz 페이지네이션 복원력**: 마지막 페이지 hang 방지 (`finviz_screener_view_resilient`)
- **하네스**: `test_hk_*`, `test_finviz_screener_pagination` 추가

### v4.4.9

- **음수 Adj Close FMS 폭증 수정**: 비양수 가격 마스킹으로 `381560.KS`류 Yahoo 글리치에서 FMS 수십~백 단위 이상치 차단

### v4.4.8

- **세부보기 Drawdown y축 통일**: 관심종목 전체 기준 고정 y-range (상단 가격 차트와 동일 정책)
- **UI**: `관심종목 초기화` 버튼 제거
- **운영 데이터 커밋**: watchlist·screened_universe는 변경 시 항상 커밋에 포함 (SSOT)

### v4.4.7

- **FMS 장기 축 6M→4M**: `R_4M`(84거래일); 게이트/quality는 복리·√t 매핑; 사전필터 Quarter/Half Up(>0%)
- **골든 순위 유지** (합성 fixture): TREND_UP > MILD_UP > FLAT > CRASHY(-999)

### v4.4.6

- **tune `fms_score` → core 파라미터 주입**: `FmsScoreParams` / `production_fms_score_params()`; 탐색은 `score_fms_from_feature_frame(..., params=...)`만 사용; `test_fms_params.py`

### v4.4.5

- **세부보기 selectbox 순정 복원**: 붙여넣기 UX용 CSS/JS·데모 스크립트 제거 (Streamlit 기본 동작)
- **푸시 전 정리**: 사전필터 실측 CSV → `scripts/fixtures/`; FMS 모듈 내부 헬퍼/가중치 단일화

### v4.4.4

- **recalib 공식 포크 제거**: `f_current`/`f_proposed` → `core.fms.score_fms_from_feature_frame`; 계약 테스트 `test_fms_recalib_parity.py`

### v4.4.3

- **core/fms 이전**: `compute_fms_snapshot` / `momentum_now_and_delta` → `core/fms.py` + 셔임; FMS 단일 소스 확정

### v4.4.2

- **core/tradeability 이전**: `calculate_tradeability_filters` → `core/tradeability.py` + 셔임; `test_tradeability.py` 추가

### v4.4.1

- **core/indicators 이전**: `ema` / `returns_pct` / `r_squared_3m` → `core/indicators.py` + `analysis_utils` 셔임; `test_indicators.py` 추가
- **사전 필터 유지 확정**: Finviz 현행 조건 유지

### v4.4.0

- **MarketDataPort/Adapter 경계 도입**: `adapters/market_data.py` (`YFinanceAdapter` / 오프라인 `FixtureAdapter`), `calculate_fms_for_batch` 다운로드/스코어 분리 및 Port 주입 지원
- **배치 log RuntimeWarning 제거**: 음수/0 가격 글리치가 R²·EMA20 로그 회귀에 유입되던 문제 가드
- 계약 테스트 추가: fixture 주입 배치 = 직접 스코어링 FMS 일치 (`tests/unit/test_market_data_port.py`)

### v4.3.1

- 배치 스캔 복구: Yahoo 레이트리밋 재시도 수정, Finviz 티커 첫글자 중복 보정, set_filter 적용
- Adj Close(배당 조정) 기준 수익률/FMS 문서화

### v4.3.0

- **Harness Engineering 도입**: FMS 스코어링을 라이브 API 없이 검증하는 테스트 하네스·문서 체계 구축
  - `compute_fms_snapshot` 공개 API, `tests/` pytest (순위·`-999`·NaN·no-network), `harness/run_fms_snapshot`
  - `HARNESS_RULES.md` / `TODO.md` / `docs/` 세션 부트스트랩, `core/`·`adapters/`·`scripts/` 스캐폴딩
  - 운영 경로(`app.py`, `run_scan_batch.py`)와 검증 자산 경계 분리 (Mock은 `tests/fixtures/`만)

### v4.2.2

- **세부 보기 차트**: Rebased(100) 시계열에 5거래일 단순이동평균선 추가(연한 회색 실선, EMA 색상 유지)

### v4.2.1

- **한국 종목명 표시 안정화**:
  - `.KQ`(코스닥) 심볼도 한국 종목으로 분류되도록 통일
  - 한국 종목명 로딩 시 `korean_universe.csv`와 `korean_etf_univers.csv`를 함께 사용
  - 한국 종목명 조회 우선순위를 `캐시 → CSV → yfinance`로 정리해 API 의존도를 최소화
- **실행 방식 단순화**:
  - `start.bat`에서 백그라운드 포트 폴링/브라우저 강제 오픈 로직을 제거하고 `streamlit run` 기본 동작으로 복원
  - `.streamlit/config.toml`의 `headless=false`로 로컬 실행 시 브라우저 자동 오픈 동작을 명확화

### v4.2.0

- **FMS 구조 확장 + 새 정답셋 기반 튜닝**:
  - 시계열 스냅샷에서 EMA20 기울기/곡률, 최근 10/5일 수익률, EMA20 아래 이탈 깊이/일수, 최근 5일 연속 하락 길이 등을 추가 피처로 추출
  - FMS를 “장·중기 추세 + EMA20 기울기/곡률 + 단기 유지/붕괴 + 리스크(DD/Vol/이탈/연속 하락)” 축으로 재설계
  - 새 정답셋 기준으로 가중치/전이폭 파라미터를 Monte Carlo로 튜닝해 inversion_rate↓, Spearman↑, pair_delta_error↓ 모두 개선

### v4.1.0

- **FMS 연속화 + 튜닝**:
  - R² 가산/추세 게이트의 임계값(0.7/0.9, 5%/8%)에서 발생하는 계단식 점프를 **smoothstep 전이**로 완화
  - Vol20 패널티 형태 튜닝 (q=70%, tail power=1.5)
  - 가중치/전이폭 **몬테카를로 튜닝**을 통해 정답셋 설명력 지표 개선
- **재보정 품질관리 강화**:
  - `fms_recalib_build_features.py` 실행 시 세션별 `__baseline_metrics.json` 자동 저장 (해당 정답셋 내부에서 current vs proposed 비교 용)

### v4.0.1

- **FMS R² 추세상승 게이트**: 평평한 그래프(R_3M/R_6M 낮음)에서 R² 가산 억제
- **FMS 검증 스크립트 단순화**: 수정 전(current) vs 수정 후(proposed)만 비교

### v4.0.0

- **새 FMS 비선형 전략 도입**:  
  - 3M/6M 수익률, 3M R², EMA50 상대위치, 조건부 1M 수익률(건강한 추세에서의 가속)을 가산하고,  
  - 최대 드로우다운, 20일 변동성, 이벤트성 1M 급등을 비선형 패널티로 감산하는 새로운 FMS 수식을 적용했습니다.
- **FMS 설명력 향상**:  
  - 과거 FMS 대비 순서쌍 역전 비율, Spearman 랭크 상관계수, 쌍별 순위차 오차 기준으로  
    사용자가 제공한 정답셋을 더 잘 설명하도록 개선했습니다.
- **FMS 재보정 기능 제품화**:  
  - UI에서 A/B 그래프 비교로 정답 순서를 수집하고,  
  - 스냅샷/세션/보조 스크립트를 이용해 새 FMS 후보를 평가·검증한 뒤에만 코드에 반영하는  
    재보정 워크플로우를 기능으로 통합했습니다.
- **세부 보기/비교 그래프 개선**:  
  - 세부보기 및 FMS 재보정 A/B 비교용 그래프를 **Rebased 100 + 로그 스케일 + 전역 고정 y축 범위**로 변경하여,  
    종목 간 기울기와 상대 상승률 패턴을 직관적으로 비교할 수 있도록 개선했습니다.

### v3.9.0

- **R² 기반 급등주 필터링**: FMS 공식에 3개월 로그 수익률의 결정계수(R²) 지표 추가
  - 높은 R²: 안정적인 우상향 추세 (매끄러운 상승 곡선)
  - 낮은 R²: 횡보 후 급등, 계단식 급등 등 불안정한 패턴 (감점 처리)
- **FMS 공식 전면 수정**: 급등주 필터링 강화를 위한 가중치 재조정
  - 1M 수익률 가중치: 0.4 → 0.2
  - 3M R² 가중치: 0.0 → 0.3 (신규)
  - 변동성 페널티: -0.4 → -0.2
- **scipy 의존성 추가**: R² 계산을 위한 scipy 라이브러리 추가

### v3.8.0

- **계좌 모드 지원**: 자유투자계좌(FREE)와 퇴직연금IRP(IRP) 모드 분리 지원
  - 각 모드별 독립적인 관심종목 관리 (watchlist_free.csv, watchlist_irp.csv)
  - 모드별 유니버스 스캔 (FREE: 미국+한국+홍콩 주식, IRP: 국내상장 ETF 전 종목)
  - 모드별 배치 스캔 결과 저장 및 로드
  - UI에서 모드 전환 시 자동으로 해당 모드의 관심종목 및 스캔 결과 표시
  - 배치 스캔 실행 시 현재 선택된 모드로 자동 실행
  - 작업 스케줄러에서 파라미터 없이 실행 시 두 모드 모두 자동 실행

### v3.7.4

- **배치 스캔 결과 메시지 UI 개선**: + 버튼 클릭 시 메시지가 넓은 폭에 표시되도록 개선
- **데이터 coverage 임계값 완화**: 신규 상장 종목이나 데이터가 일부 없는 종목도 포함 가능하도록 개선
- **당일 고가/저가 0 처리**: 국장 개장 시간 Yahoo Finance 데이터 이슈 대응 (전일 데이터로 대체)

### v3.7.3

- **거래 적합성 필터 디버그 로깅 확대**: 모든 국가 종목(-999 발생 종목)에 대해 디버그 정보 표시
- **디버그 로그 UI 개선**: 실격된 종목 정보를 요약 테이블과 상세 정보로 더 읽기 쉽게 표시

### v3.7.2

- **거래 적합성 필터 디버그 로깅**: 오전 장 시작 시간 국내 종목 -999 문제 진단을 위한 상세 디버그 정보 제공

### v3.7.1

- **레버리지/인버스 ETF 필터링 강화**: 배치 스캔 시 레버리지/인버스 ETF 탐색 제외 기능 개선 (LLYX, SMST, GGLL, GOOX 등 패턴 자동 감지)
- **관심종목 추가 실패 시 피드백 개선**: 배치 결과에서 관심종목 추가 시 실패 이유 명확히 표시
- **수동 추가 기능 개선**: 수동 관리에서 어떤 종목이든 추가 가능하도록 제한 제거 (레버리지/인버스 ETF 포함)

### v3.7.0

- **동적 데이터 기간 계산**: 사용자 선택 옵션에 따라 필요한 최소 데이터 기간만 동적으로 계산
- **차트 기간 버그 수정**: 1M 선택 시 한국 종목 데이터 부족 문제 해결
- **세부 보기 차트 기간 수정**: 선택된 차트 기간에 맞춰 표시
- **비교 차트 로그 스케일 고정**: 항상 로그 스케일로 표시
- **UI 개선**: 불필요한 표시 제거 및 레이아웃 정리

### v3.6.3

- **차트 기간 옵션 개선**: 1M 옵션 추가, 5Y 옵션 제거
- **표시 종목 수 옵션 확장**: 최대 60개까지 선택 가능
- **버전 일관성 개선**: UI와 코드 버전 동기화

### v3.6.2

- **종목명 조회 신뢰성 향상**: 한국 상장 종목이 유니버스 파일에 없더라도 캐시 → 한국 유니버스 → yfinance 순으로 이름을 탐색하고, 성공한 결과를 캐시에 저장해 재호출 시 API 비용을 최소화

### v3.6.1

- **세부 보기 섹션 개선**: `수익률–변동성 이동맵` 위에 배치해 흐름 개선, 정렬 순서에 맞춘 두 자리 인덱스(`[01]`) 표시로 빠른 식별 지원
- **빈 결과 처리 강화**: 세부 보기 목록이 비었을 때 안전하게 안내 메시지를 표시하고 예외를 예방

### v3.6.0

- **한국 종목명 파일 기반 관리**: `korean_universe.csv`에 `Name` 컬럼 추가로 한국 종목명 직접 관리
- **종목명 영구 캐시 시스템**: `symbol_names_cache.json` 파일 기반으로 비한국 종목명 캐시 저장
- **한국 종목명 처리 효율화**: 한국 종목(.KS)은 yfinance API 호출 없이 파일에서 직접 읽어오도록 개선
- **한국 종목명 정확도 향상**: yfinance의 이상한 문자열 대신 파일에 입력한 정확한 종목명 사용

### v3.5.0

- **수익성-변동성 그래프 로그 스케일**: 세로축에 로그 스케일 적용으로 넓은 범위의 수익률 비교 용이
- **애니메이션 모드 종목명 표시**: 애니메이션 중에도 종목명 표시로 가독성 향상
- **버그 수정**: 데이터 부족 시 발생할 수 있는 오류 방지

### v3.4.0

- **세부보기 종목 네비게이션 기능**: 정렬된 순서로 종목 탐색 (이전/다음/맨 끝 이동)
- 코드 정리 및 파일 정리

### v3.3.0

- **배치 스캔 결과 UI 복원**: 배치 스캔 결과를 UI에서 확인하고 관심종목에 추가하는 기능 추가
- FMS 순 정렬 및 페이징으로 대량 결과 효율적 탐색
- 원클릭 관심종목 추가로 빠른 포트폴리오 구성

### v3.2.0

- **FMS 계산 Z-score 왜곡 문제 해결**: -999 패널티 종목을 Z-score 계산에서 제외하여 정확도 향상
- **배치 간 API 제한 방지**: chunk 간 0.1초 대기 추가로 yfinance 레이트리밋 대응 강화
- **Watchlist 실격 필터링**: 배치 스캔 시 참조 데이터 품질 향상
- **아키텍처 단순화**: 인터랙티브 스캔 제거, 배치 스캔 전용 시스템으로 전환
- **코드 중복 제거**: `run_scan_batch.py`에서 중앙화된 `analysis_utils.py` 로직 사용
- **FMS 임계값 조정**: 신규 탐색 시 2.0→0.0으로 조정하여 더 많은 종목 포함
- **다국가 통합 스캔**: 미국(Finviz) + 한국(KOSPI200/KOSDAQ150) 유니버스 병합으로 글로벌 모멘텀 종목 탐색

### v3.0.8

- **치명적 변동성 필터 로직 개선**: True Range 기반 계산으로 가격 갭 반영
  - 기존: `(당일 고가 - 당일 저가) / 전일 종가` (일중 변동성만 측정)
  - 신규: `max(고가-저가, |고가-전일종가|, |저가-전일종가|) / 전일종가` (가격 갭 포함)
  - 주말/휴일 등 비거래일에 발생하는 큰 폭의 가격 변동(갭 상승/하락) 정확 감지
  - 업계 표준 True Range 방식 적용으로 필터링 정확도 대폭 향상
- **필터링 임계값 완화**: 치명적 변동성 필터를 15%에서 30%로 완화하여 더 많은 종목이 통과할 수 있도록 개선
- **필터링 상태 표시**: 모멘텀 테이블에 Filter_Status 컬럼 추가로 실격 이유를 명확히 표시

### v3.0.7

- **FMS 계산 로직 통일**: 변동성 가속도 제거로 일관된 FMS 공식 적용
- **거래 적합성 필터**: True Range 기반 치명적 변동성 및 반복적 하방리스크 감지로 부적합 종목 자동 실격
- **동적 컬럼 재정렬**: FMS 전략에 맞춰 결과 테이블 컬럼 순서 자동 조정
- **문서 일관성**: config.py, README.md의 FMS 설명 통일

### v3.0.6

- **FMS 계산 로직 일관성 개선**: Z-score 정규화 기준을 관심종목 집합으로 통일
- **참조 데이터 시스템 도입**: 유니버스 스캔과 관심종목에서 동일한 FMS 값 계산
- **신뢰성 향상**: FMS 값 불일치 문제 해결로 사용자 경험 개선
- **코드 정리**: 사용하지 않는 함수 제거 및 최적화

### v3.0.5

- **FMS 전략 단일화**: Standard 전략 제거, 안정 성장형 전략으로 통일
- **FMS 계산 로직 일관성**: 스캔 결과와 관심종목 표시에서 동일한 FMS 값 계산
- **UI 단순화**: FMS 전략 선택 UI 제거, 도구 및 도움말에만 수식 설명 표시
- **페이징 버튼 중복 방지**: 연속 클릭 시 화살표 버튼 중복 생성 버그 수정
- **캐시 초기화 메시지 개선**: 불필요한 "Rerun 클릭" 메시지 제거

### v3.0.4

- **신규 FMS 전략 '안정 성장형' 추가**: 추세의 지속성과 안정성을 중시하는 새로운 FMS 계산 방식
- **3M수익률 및 변동성 가속도 지표**: 중기 모멘텀 평가 및 급등 패턴 감지
- **FMS 전략 선택 UI**: 좌측 사이드바에서 Standard/Stable Growth 전략 선택 가능
- **이벤트성 급등주 필터링**: 변동성 가속도로 수직 폭등 종목 자동 제거

### v3.0.3

- **모듈화된 유니버스 관리**: `universe_utils.py`로 유니버스 관리 로직 분리
- **실시간 진행률 표시**: 유니버스 업데이트 및 FMS 스캔 과정의 실시간 진행률 표시
- **유니버스 파일 업로드**: CSV 파일을 통한 유니버스 교체 기능 복구
- **성능 최적화**: 중복 표시 제거 및 사용자 경험 개선

### v2.9.1

- **레버리지/인버스 ETF 제외**: 모멘텀의 본질을 흐리는 레버리지형 ETF 자동 제외
- **다양한 레버리지 패턴 인식**: 2X, 3X, 2배, 3배, Inverse, Short, Bear 등 다양한 패턴 지원
- **모멘텀 분석 정확도 향상**: 순수한 모멘텀 종목만 선별하여 분석 품질 향상

### v2.9

- **진정한 전체 시장 스캔**: Finviz.com 기반 실시간 유니버스 스크리닝 시스템
- **2단계 추진 로켓 방식**: 사전 필터링 + 온디맨드 FMS 스캔으로 성능 최적화
- **유니버스 스크리닝 스크립트**: `update_universe.py`로 매일 업데이트되는 유망주 목록 생성
- **동적 스크리닝**: 가격 $5+, 거래량 200K+, 1개월 수익률 0%+ 등 다중 필터 적용
- **대규모 데이터 처리**: 8,000+ 종목에서 유망주 후보군 자동 발굴

### v2.8

- **관심종목 영구 저장**: CSV 파일을 통한 관심종목 영구 저장
- **관심종목 관리 UI**: 사이드바에 직관적인 관심종목 관리 인터페이스
- **진부한 종목 자동 편출**: FMS 기반 저성과 종목 제거 제안
- **동적 포트폴리오 관리**: 실시간 관심종목 추가/삭제
- **모듈화**: 관심종목 관리 로직을 별도 모듈로 분리
- **함수 정의 순서 최적화**: 의존성 순서에 따른 함수 재정렬로 안정성 향상

### v2.7

- FMS 설명 박스 추가로 사용자 이해도 향상
- 애니메이션 자동 재생 및 꼬리 길이 자동 조정
- 시각적 개선 (과거 시점과 꼬리 중복 제거)
- UI 간소화 (상단 KPI 제거, 기본값 최적화)

### v2.6

- FMS 기반 모멘텀 스코어링
- 다국가 시장 통합 분석
- KRW 환산 가격 비교

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ 면책 조항

이 도구는 교육 및 연구 목적으로만 제공됩니다. 투자 결정에 사용하기 전에 반드시 전문가의 조언을 구하시기 바랍니다. 과거 성과가 미래 결과를 보장하지 않습니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**⚡ KRW Momentum Radar** - 모멘텀의 힘을 시각화하세요!
