# FMS 재보정 워크플로우

> 최종 갱신: 2026-07-29 (KST)
> 현재 상태: 최신 80종 정답셋의 **원점 재피팅 후보가 v4.6.0 production으로 승격 완료**

이 문서는 사용자가 A/B 차트 비교로 만든 순위를 바탕으로, 그래프에서 설명 가능한
피처를 발견하고 FMS를 원점부터 다시 피팅하는 현재 표준 절차를 정의한다.

---

## 1. 핵심 원칙

### 1.1 진실과 오류 수정

- 실제 동작 확인은 **소스코드가 최우선 진실**이다.
- 우선순위는 `실제 동작 소스코드 → .cursorrules → HARNESS_RULES.md / docs`다.
- 다만 소스에서 부호, 뺄셈 순서, 계산창 등 명백한 의미 오류가 발견되면
  현재 동작을 정확히 기록한 뒤 **피처 의미에 맞게 수정하고 테스트한다**.
- 문서와 과거 수식은 보존 대상이 아니라 재현·검증을 위한 근거다.

### 1.2 정답의 의미

- 정답은 사용자가 고정된 차트를 보고 결정한 **서열(rank)** 이다.
- 등수 간 간격이 동일하다고 가정하지 않는다.
- 학습 목표는 점수 회귀가 아니라 “A가 B보다 위”라는 **pairwise ordering**이다.
- 사용자 메모는 피처 아이디어의 원천이지만, 설명과 실제 정렬이 충돌하면
  **최종 정렬 결과를 우선**한다.
- 재검토에서 판단이 바뀐 쌍은 오류로 삭제하지 않고 label uncertainty로 검증한다.

### 1.3 최신 정답셋 하나만 사용

- 피팅·모델 선택·평가에는 JSON `saved_at`이 가장 최신인
  `phase == "done"` 완료 세션 **하나만** 사용한다.
- 파일명 순서나 filesystem mtime으로 최신 세션을 선택하지 않는다.
- 과거 세션은 합치거나 외부 검증셋으로 재사용하지 않는다.
- 현재 기준 정답셋:
  - session: `20260729_recent1m`
  - snapshot: `fms_20260729_154752`
  - symbols: 80
  - chart period: `3M`
  - review inconsistencies: 3

### 1.4 데이터와 평가 누수 금지

- A/B 비교 시작 시 가격 데이터를 한 번 고정하고 세션 종료까지 갱신하지 않는다.
- 정규화 통계, 결측 대체값, 상관 피처 제거, L1 선택은 각 학습 fold 안에서만 계산한다.
- 모델에 종목명, 국가, 개별 종목 예외 규칙을 넣지 않는다.
- 최신 정답셋의 20%를 rank-stratified audit symbols로 잠그고, development 80%에서
  피처 발견·모델 선택·잔차 분석을 끝낸 후 audit을 한 번만 평가한다.
- audit 결과를 보고 수식을 변경하면 그 audit은 더 이상 미관측 검증셋이 아니다.
  변경 후보는 nested holdout으로만 평가하고, 새 정답셋에서 다시 외부 검증해야 한다.

### 1.5 해석 가능성과 production 경계

- 후보는 UI에 표시된 가격·EMA·Drawdown에서 설명 가능한 피처만 사용한다.
- production FMS는 **benchmark일 뿐**, 원점 피팅의 입력·초기값·가산항이 아니다.
- 원점 후보 점수는 반드시 0에서 시작한다.
- 거래적합성 `FMS=-999`는 raw OHLC 기반 안전정책으로, 가격 순위 피팅과 분리한다.
- 후보 보고와 production 승격은 별도 단계다. 사용자 컨펌 전에는
  `core/fms.py`, 앱, 배치, 버전을 변경하지 않는다.

---

## 2. 현재 구현 구조

### 정답셋·스냅샷

- `calibration/session.py`
  - 세션 저장·복원, merge sort, 10% 인접 재검토
  - `latest_completed_session()`은 JSON `saved_at` 기준으로 최신 완료 세션 선택
- `calibration/manifest.py`
  - session/snapshot ID, 80종 순서, ranking hash, prices hash 기록
  - development/audit symbol split 고정
- `fms_recalib_manifest.json`
  - 현재 실행에 사용한 재현 manifest

### 피처

- `core/fms_features.py`
  - 네트워크 없는 순수 DataFrame 입력형 feature builder
  - 3M visible window 기반 후보 피처
  - 피처 방향성(`FEATURE_DIRECTION`)과 reference normalization 헬퍼
- `calibration/fms_recalib_build_features.py`
  - 최신 manifest와 snapshot을 검증하고 `fms_recalib_features.csv` 생성
  - production baseline metrics 저장

### 원점 피팅·검증

- `calibration/fms_recalib_refit.py`
  - sparse linear / monotone GAM / limited interaction 비교
  - pairwise loss, nested holdout, bootstrap, LOO, label variants
- `calibration/ranking_metrics.py`
  - inversion, Spearman, pair-delta, top-quintile recall
- `calibration/fms_recalib_plot_residuals.py`
  - development의 큰 역전 종목을 고정 3M 차트로 출력

### 생성되는 산출물

- `fms_recalib_features.csv` (build 실행 시)
- `fms_recalib_scratch_candidate.json`
- `fms_recalib_scratch_scores.csv`
- `fms_recalib_scratch_residual_pairs.csv`
- `fms_recalib_scratch_residual_charts.png`

`fms_recalib_latest_fit.json`과 과거 `production_fms + addon` 결과는
incremental 실험 기록일 뿐, 전면 재피팅 후보가 아니다.

---

## 3. 1단계 — UI에서 정답셋 수집

### 3.1 새 세션 시작

1. 앱을 실행한다.
2. 본문의 **FMS 재보정** 섹션으로 이동한다.
3. 새 세션을 시작해 현재 가격 패널을 snapshot으로 고정한다.
4. 세션 ID와 snapshot ID를 확인한다.

### 3.2 차트 비교 기준

- 가격은 시작값 100으로 리베이스한 로그 스케일을 사용한다.
- A/B 차트의 y축 범위를 동일하게 유지한다.
- 사용자는 “오늘 이후 상승추세가 얼마간 지속될 가능성”을 기준으로 선택한다.
- 비교 시 주로 관찰하는 패턴:
  - 최근 2~3주의 기울기와 매끄러움
  - 최근 3일의 추세 훼손·회복
  - 이전 추세가 최근 상승을 지지하는 정도
  - EMA20 위치·기울기·곡률과 미회복 기간
  - 하락 변동성, drawdown, 회복 속도
  - 정체 기간과 고점 이후 동력 소진
  - 급등 자체가 아니라 급등 후 진행·정체·급락 경로

### 3.3 정렬과 재검토

- merge sort 상태 머신으로 필요한 비교만 수행한다.
- 비교 직후 상태를 저장해 중단 후 재개할 수 있어야 한다.
- 정렬 완료 후 인접 쌍 약 10%를 다시 비교한다.
- 최초·재검토 선택이 다르면 `inconsistencies`에 두 판단을 모두 기록한다.
- review queue가 끝나 `phase == "done"`이고 `final_ranking`이 있어야 정답셋으로 인정한다.

---

## 4. 2단계 — manifest와 피처 테이블 고정

프로젝트 루트에서:

```bash
python fms_recalib_build_features.py
```

이 명령은 다음을 수행한다.

1. `saved_at` 기준 최신 완료 세션 하나 선택
2. ranking과 snapshot columns의 exact alignment 확인
3. ranking/prices hash와 audit split 기록
4. 3M visible-window 피처 생성
5. production FMS를 benchmark로만 계산

### 4.1 visible-window 정책

- 이번 세션의 차트 기간은 3M이므로 후보는 원칙적으로 63거래일 안에서 계산한다.
- `R_4M`은 production baseline scorer 호환을 위해 테이블에 있을 수 있지만
  원점 후보 목록에서는 제외한다.
- `MaxDD_Pct` 후보는 전체 history가 아니라 visible 3M window에서 계산한다.
- 미래 데이터, live API 갱신, 사용자가 보지 못한 외부 정보는 사용하지 않는다.

### 4.2 피처 카탈로그

초기 목록은 확정 수식이 아니라 탐색 후보군이다.

- 최근성: 3/5/10/15/21/42/63일 수익·로그기울기·R²
- 추세 질: slope×R², trend efficiency, monotonicity
- EMA: EMA20/50 위치, 기울기, 곡률, 이탈 깊이·일수·연속기간
- 회복: 최근 조정 후 3일 회복, drawdown 회복률, 재돌파
- 하방 위험: downside RMS, 최악 일수익, 연속 하락
- 비대칭성: upside/downside RMS와 변동성 비대칭
- 정체: 고점 이후 경과일, range compression, stale age
- 급등 후 경로: post-spike stall, jump discontinuity
- 연속성: 최근 상승 streak, 15일 positive efficiency
- 제한 상호작용: 최근 추세×과거 신뢰도, 회복×하방위험,
  급등×follow-through

각 피처에는 다음이 명시되어야 한다.

- 자연어 의미
- 계산창과 단위
- 좋은 방향/나쁜 방향
- 결측 처리
- clip·정규화 정책
- synthetic fixture에서 기대되는 동작

---

## 5. 3단계 — development에서 피처 발견

### 5.1 먼저 자연어 패턴을 설명

- 상·중·하위 그룹의 공통 패턴을 정리한다.
- 사용자 메모를 기계적으로 수식화하지 않고 실제 rank와 대조한다.
- 같은 직관을 여러 수치 표현으로 만들되, 상관 피처는 대표 하나만 남긴다.

### 5.2 잔차 차트 반복

1. development에서 초기 모델을 피팅한다.
2. 큰 rank-gap 역전, 상위 false negative/positive를 추출한다.
3. snapshot의 동일한 3M 리베이스 가격·EMA 차트를 직접 확인한다.
4. “왜 기존 피처가 이 패턴을 구분하지 못했는가”를 자연어로 기록한다.
5. 여러 종목에 반복되는 패턴만 새 피처로 일반화한다.
6. synthetic 의미 테스트를 추가하고 nested holdout을 다시 실행한다.

2026-07-29 잔차 검토에서는 다음을 추가했다.

- `JUMP_DISCONTINUITY_3M`
  - 상승분이 짧은 급등에 집중되고 최근 20일 follow-through가 약할수록 증가
  - 높은 값은 감점 방향
- `RECENT_3D_VS_21D_TREND`
  - 최근 3일 일평균 로그수익과 21일 로그기울기의 차이
  - 최근 추세 훼손·회복 후보

BLFS·LXP 유형의 과대평가는 완전히 제거되지 않았으므로 잔여 한계로 보고한다.

---

## 6. 4단계 — 원점 모델 피팅

### 6.1 전처리

각 학습 fold 안에서만:

1. 결측·무한값을 train median으로 대체
2. train mean/std로 Z-score
3. 극단값을 `[-4, 4]`로 clip
4. 감점 피처는 부호를 반전해 “값이 높을수록 좋은 축”으로 통일
5. 상관계수 절댓값 0.94 이상인 후보는 rank 연관성이 높은 대표만 유지

test/audit에는 train에서 얻은 median/mean/std를 그대로 적용한다.

### 6.2 pairwise 가중치 피팅

후보 점수는 0에서 시작한다.

```text
score_i = Σ(w_k × signed_Z(feature_i,k))
```

development에서 정답상 A가 B보다 위인 모든 쌍에 대해 다음을 최소화한다.

```text
mean[log(1 + exp(-(score_A - score_B)))]
+ λ × Σw
+ 0.002 × Σw²
```

- `0 ≤ w ≤ 6`: 방향성을 보존하는 비음수 제약
- optimizer: SciPy L-BFGS-B
- L1 후보: `0.01, 0.03, 0.07, 0.15`
- 최대 항 수: 10
- rank gap으로 pair에 추가 가중치를 주지 않는다.

### 6.3 비교 모델군

1. `sparse_linear`
   - signed Z 피처의 희소 가중합
2. `monotone_gam`
   - linear / tanh / softplus 연속 단조 basis
3. `limited_interactions`
   - 사용자 직관으로 사전 정의한 소수의 연속 confirmation interaction

블랙박스 tree/boosting 모델과 종목별 규칙은 사용하지 않는다.

### 6.4 모델 선택

- development에서 repeated 5-fold outer validation을 수행한다.
- 각 outer train 내부 4-fold에서 L1을 선택한다.
- 주 지표는 inversion, 보조 지표는 Spearman과 pair-delta다.
- raw best보다 inversion 1 standard error 이내인 모델 중 가장 단순한 모델을 선택한다.
- family와 L1을 동결한 뒤 development 전체에서 최종 가중치를 적합한다.

---

## 7. 5단계 — 과적합 검증

### 7.1 필수 검증

- repeated nested symbol holdout
- symbol bootstrap stability selection
- leave-one-symbol-out
- 상·중·하위 구간 오차
- top-quintile recall
- feature-family·변환 ablation

개별 pair를 독립 표본처럼 bootstrap하지 않는다. symbol을 복원추출하고,
동일 symbol이 여러 번 뽑힌 빈도를 bootstrap sample에 반영한다.

### 7.2 label uncertainty

재검토 불일치가 `k`개면 각 쌍의 최초/재검토 판단을 독립 조합해
`2^k`개 순위 변형을 평가한다.

현재 세션은 불일치 3개이므로 **8개 변형**을 모두 평가한다.

### 7.3 부분집합 지표

holdout·audit의 pair-delta와 Spearman을 계산할 때 원래 1~80 rank를 그대로
사용하지 않는다. 해당 부분집합 내부 순서를 연속 `1…n`으로 다시 부여한다.

### 7.4 audit

- audit symbols는 모델 구조·피처·가중치를 동결한 후 한 번만 평가한다.
- audit의 세 주 지표가 production benchmark보다 악화되면 승격하지 않는다.
- audit 결과를 본 뒤 수정한 후보는 같은 audit으로 “미관측 검증 완료”라고 부르지 않는다.

#### 현재 세션의 검증 한계

2026-07-29 작업 중 첫 후보에서 audit 결과를 확인한 후, **development 잔차만을
근거로** 신규 피처를 추가하고 audit을 다시 계산했다. 신규 피처 선택에 audit
수치를 사용하지는 않았지만, 최종 후보의 audit은 엄밀한 의미의 완전 미관측
검증이라고 주장하지 않는다. 최종 외부 검증은 다음 신규 정답셋에서 수행해야 한다.

---

## 8. 평가 지표와 채택 기준

### 지표

- `inversion_rate`: 전체 쌍 중 정답과 순서가 뒤집힌 비율, 낮을수록 좋음
- `spearman_rho`: 정답 rank와 모델 rank의 상관, 높을수록 좋음
- `pair_delta_error`: 두 종목 간 정답 순위차와 모델 순위차의 평균 절대오차
- `top_quintile_recall`: 정답 상위 20%를 모델 상위 20%가 포함한 비율

### 후보 보고 조건

- full set과 development에서 production 대비:
  - inversion 감소
  - Spearman 증가
  - pair-delta 감소
- nested holdout에서 안정적인 개선
- 핵심 피처의 bootstrap 방향·선택 안정성
- LOO에서 특정 종목 하나 제거로 결론이 뒤집히지 않음
- 모든 label variants에서 개선 방향 유지
- 수식과 각 피처를 자연어로 설명 가능

한 개 정답셋에서 수치가 좋아도 미래 일반화를 증명한 것은 아니다.

---

## 9. 승격된 원점 모델 — 2026-07-29

이 절은 실행 결과 스냅샷이며 production 수식이 아니다.

### 선택 결과

- family raw best: limited interactions
- one-standard-error 선택: **sparse linear**
- 상태: `promoted_to_production`
- 승격 버전: `v4.6.0`

```text
+ 0.846427 × Z(R2_3M)
+ 0.601307 × Z(DD_RECOVERY)
+ 0.354317 × Z(TREND_QUALITY_21D)
- 0.279017 × Z(JUMP_DISCONTINUITY_3M)
- 0.196604 × Z(UNDER_EMA20_DAYS)
+ 0.186983 × Z(R_3M)
- 0.181753 × Z(STALE_AGE)
+ 0.107915 × Z(UP_STREAK_5D)
+ 0.107766 × Z(TREND_EFFICIENCY_REWARD_15D)
- 0.104169 × Z(RANGE_COMPRESSION_20D)
```

### 최신 80종 전체 지표

| 지표 | production v4.5.1 | 원점 후보 |
|---|---:|---:|
| inversion | 0.2373 | 0.1177 |
| Spearman | 0.7024 | 0.8917 |
| pair-delta | 19.8032 | 10.4677 |
| top 20% recall | 0.6875 | 0.8750 |

### audit 16종

| 지표 | production v4.5.1 | 원점 후보 |
|---|---:|---:|
| inversion | 0.2083 | 0.1417 |
| Spearman | 0.7118 | 0.8000 |
| pair-delta | 3.7667 | 2.7333 |

- label variants: 8/8에서 production 대비 세 지표 동시 개선
- LOO inversion range: 0.1126~0.1254
- 핵심 bootstrap 선택률:
  - R2_3M: 100%
  - DD_RECOVERY: 95.8%
  - JUMP_DISCONTINUITY_3M: 86.7%
  - TREND_QUALITY_21D: 78.3%
  - R_3M: 75.0%

하위 가중치 피처의 선택률은 더 낮으므로 승격 전 수식 단순화 여부를 함께 검토한다.

---

## 10. Monte Carlo의 역할

- 현재 sparse pairwise loss는 같은 변수·제약 아래 거의 볼록이므로
  광범위한 무작위 탐색을 주 optimizer로 쓰지 않는다.
- Monte Carlo는 수식 구조를 정한 뒤 다음 용도로만 선택적으로 사용한다.
  - 현재 가중치 주변 민감도·평탄성 확인
  - 실제 rank 지표의 constrained local refinement
  - weight=0 ablation과 임계값 주변 탐색
- 전체 배율은 순위에 영향이 없으므로 `Σw=1` 등으로 scale을 고정한다.
- 방향·해석 가능성·sparsity를 유지하고 각 inner train 안에서만 탐색한다.
- audit을 확인한 뒤 Monte Carlo로 수정한 후보는 같은 audit으로 재승인하지 않는다.

---

## 11. 후보 보고와 production 승격

### 11.1 후보 보고

다음을 사용자에게 보고한다.

- 자연어 동작 설명
- 전체 수식과 정규화 정책
- 선택·제거된 기존/신규 피처
- production 대비 full/development/audit 지표
- nested/bootstrap/LOO/label variant 결과
- 가장 큰 개선·악화 사례
- 단일 정답셋과 audit 사용 이력 등 한계

여기서 사용자 컨펌을 기다린다.

2026-07-29 모델은 위 보고 후 사용자 컨펌을 받아 v4.6.0으로 승격했다.
이 문장의 승격 이력은 다음 재보정의 후보-승격 분리 원칙을 완화하지 않는다.

### 11.2 컨펌 후에만 승격

승인 후:

1. 공유 feature builder와 score body를 `core/` SSOT로 구현
2. `compute_fms_snapshot`과 feature-frame scorer의 parity 보장
3. 앱·배치가 동일 core 경로를 사용하도록 연결
4. 거래적합성 `-999` 정책 유지
5. 골든 순위, 경계 연속성, 결측·비양수 가격, 네트워크 금지 테스트
6. 전체 pytest, FMS harness, app/batch import smoke 실행
7. 버전과 `app.py`, `config.FMS_FORMULA`, README, CHANGELOG, TODO,
   HARNESS_RULES, 본 문서 동기화

컨펌 전에는 version bump나 production 공식 변경을 하지 않는다.

---

## 12. 재실행 명령

```bash
# 1. 최신 완료 세션과 snapshot으로 manifest/features 생성
python fms_recalib_build_features.py

# 2. 0점에서 원점 재피팅 + 검증
python fms_recalib_refit.py

# 3. development 잔차 차트
python -m calibration.fms_recalib_plot_residuals

# 4. 전체 회귀
python -m pytest
python -m harness.run_fms_snapshot
```

### 다음 작업을 AI에게 지시하는 예시

> 최신 완료 FMS 재보정 세션 하나만 사용해 원점 재피팅을 수행해.
> production FMS는 benchmark로만 쓰고 후보 점수는 0에서 시작해.
> development에서 피처 발견·잔차 차트 검토·nested validation을 끝낸 후
> audit을 한 번만 평가해. sparse/GAM/제한 상호작용을 비교하고
> one-standard-error 규칙을 적용해. bootstrap, LOO, 모든 review label
> variants를 검증한 뒤 후보 보고 단계에서 멈추고 production은 수정하지 마.

---

## 13. 레거시 도구의 위치

다음 도구는 과거 incremental 수식 평가 또는 특정 항 미세조정용이다.

- `fms_recalib_evaluate_formulas.py`
- `fms_recalib_rank_metrics.py`
- `fms_recalib_tune_vol_penalty.py`
- `fms_recalib_tune_weights_and_transitions.py`
- `calibration/fms_recalib_fit_latest.py`
- `calibration/fms_recalib_screen_short_horizon.py`

이들을 전면 재피팅의 기본 경로로 사용하지 않는다. 기존 production 구조를 유지한 채
항을 더하는 결과는 **incremental 실험**으로 명시하며 원점 재피팅으로 보고하지 않는다.
