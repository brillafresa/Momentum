# FMS 재보정 워크플로우

> 최종 갱신: 2026-08-02 (KST)  
> 현재 production: **v5.0.0** (`alive_pullback` 절대 비선형)  
> 원점 재피팅 표준: **자연어 규칙 → 비중첩/고해상도 피처 → 비선형 수식 → 몬테카를로 경쟁**

이 문서는 사용자가 A/B 차트 비교로 만든 순위를 바탕으로 FMS를 **원점부터**
다시 피팅하는 현재 표준 절차다. 문서와 이번 지시서가 충돌하면 **지시서(및 본
문서의 최신 절)** 을 따른다. 과거 sparse-linear/L-BFGS 중심 절차는 레거시로
보존하되, 신규 재피팅의 기본 경로가 아니다.

Production v5.0.0은 **절대 경로 점수**다(watchlist 상대 Z 없음).
`reference_prices_krw` API는 호환을 위해 남지만 점수에 영향을 주지 않는다.
레거시 sparse+상대Z는 `score_legacy_sparse_fms_features` 하네스 전용이다.

---

## 1. 핵심 원칙

### 1.1 진실과 오류 수정

- 실제 동작 확인은 **소스코드가 최우선 진실**이다.
- 우선순위는 `실제 동작 소스코드 → .cursorrules → HARNESS_RULES.md / docs`다.
- 다만 소스에서 부호, 뺄셈 순서, 계산창 등 명백한 의미 오류가 발견되면
  현재 동작을 정확히 기록한 뒤 **피처 의미에 맞게 수정하고 테스트한다**.

### 1.2 정답의 의미

- 정답은 사용자가 고정된 차트를 보고 결정한 **서열(rank)** 이다.
- 학습 목표는 점수 회귀가 아니라 “A가 B보다 위”라는 **pairwise ordering**이다.
- 설명과 실제 정렬이 충돌하면 **최종 정렬 결과**를 우선한다.
- 재검토 불일치는 삭제하지 않고 label uncertainty로 검증한다.

### 1.3 최신 정답셋 하나만 사용

- 피팅·모델 선택·평가에는 JSON `saved_at`이 가장 최신인
  `phase == "done"` 완료 세션 **하나만** 사용한다.
- 과거 세션을 합치거나 외부 검증셋으로 재사용하지 않는다.
- 현재 기준 정답셋 (2026-08-02):
  - session: `cal_fms_20260730_190637`
  - snapshot: `fms_20260730_190637`
  - symbols: 147
  - chart period: `3M`
  - review inconsistencies: 5

### 1.4 데이터와 평가 누수 금지

- A/B 비교 시작 시 가격 패널을 스냅샷으로 고정한다.
- 정규화·결측 대체·파라미터 선택은 각 학습 fold / development 안에서만 한다.
- **종목명·국가·자산군 예외 규칙을 모델에 넣지 않는다.**
  (채권·예금 ETF 문제는 절대수익·변동성 등 일반 피처로 해결한다.)
- 정답셋의 20%를 rank-stratified audit으로 잠그고, development에서 피처·수식·
  파라미터 선택을 끝낸 뒤 audit을 **한 번만** 평가한다.

### 1.5 해석 가능성과 production 경계

- 후보는 가격·EMA·Drawdown에서 설명 가능한 피처/수식만 사용한다.
- production FMS는 **benchmark일 뿐**, 원점 후보의 입력·초기값·가산항이 아니다.
- 원점 후보 점수는 0에서 시작하는 **독립 비선형 수식**이다.
- 거래적합성 `FMS=-999`는 가격 순위 피팅과 분리한다.
- 사용자 승인 전에는 `core/fms.py`·앱·배치·버전을 변경하지 않는다.

### 1.6 모델링 4단계 (신규 표준 — 필수)

전통적 피처 셀렉션/선형 회귀만으로 끝내지 않는다. 아래 순서를 지킨다.

1. **패턴 분석 → 자연어 규칙**  
   정렬된 그래프/그룹 통계를 보고 사람이 쓰는 문장으로 규칙을 먼저 적는다.
2. **피처 엔지니어링**  
   최근일수록 높은 해상도 + **비중첩(non-overlapping) 구간** 피처를 반드시 포함.
3. **비선형 수식 고안**  
   자연어 규칙을 반응 함수(제곱, sqrt, log, softplus, regime switch 등)로 표현.
4. **몬테카를로 적합 + 경쟁 평가**  
   여러 비선형 family의 파라미터를 MC로 샘플링·정제하고 서로 경쟁시킨다.

---

## 2. 현재 구현 구조

### 정답셋·스냅샷

- `calibration/session.py` — merge sort, `saved_at` 기준 최신 완료 세션
- `calibration/manifest.py` — ranking/prices hash, development/audit split
- `fms_recalib_manifest.json`

### 피처

- `core/fms_features.py`
  - visible 3M(63d) 피처 + **SEG_*** 비중첩 구간(0–3, 3–5, 5–10, 10–21, 21–63 및 0–5/5–21/21–63)
  - `PRIOR_SUPPORT_SIGN` (이전 구간 상승 여부)
- `calibration/fms_recalib_build_features.py` → `fms_recalib_features.csv`

### 패턴·자연어

- `calibration/fms_recalib_inspect_patterns.py`
  - TOP/MID/BOT 요약 + `fms_recalib_natural_language_rules.json`

### 원점 피팅 (기본 경로)

- `calibration/nonlinear_formulas.py` — 해석 가능 비선형 family
- `calibration/fms_recalib_nonlinear_mc.py` — MC 탐색·경쟁·검증
- 산출: `fms_recalib_scratch_candidate.json`, `*_scores.csv`, `*_residual_pairs.csv`

### 레거시 (비기본)

- `calibration/fms_recalib_refit.py` — sparse linear / monotone GAM / limited interaction + L-BFGS  
  비교·회귀용으로 유지. **신규 재피팅의 기본 경로가 아니다.**

### 지표·잔차

- `calibration/ranking_metrics.py`
- `calibration/fms_recalib_plot_residuals.py`

---

## 3. 1단계 — UI에서 정답셋 수집

1. Streamlit 앱 실행 → **차트 기간 3M** 확인.
2. **FMS 재보정**에서 새 세션 시작(스냅샷 고정).
3. “오늘 이후 상승이 지속될 가능성” 기준으로 A/B 선택.
4. 정렬 + 인접 재검토(~10%) 후 `phase == "done"`.

관찰 포인트: 최근 3일·1주·2주, 이전 추세의 지지, 급등 후 정체, 절대 저수익 경로.

편향 완화: 관심목록만으로 정답셋을 만들지 말고, 저수익 채권·예금성 ETF와
임의 종목을 충분히 포함해 **공통 편향(예: 절대수익 하한)** 이 학습되도록 한다.

---

## 4. 2단계 — manifest·피처 테이블

```bash
python fms_recalib_build_features.py
```

- 최신 완료 세션 하나 + snapshot alignment
- development/audit split 고정
- production FMS는 baseline metrics만 기록

### 4.1 Visible-window

- 차트 3M → 후보 피처는 원칙적으로 63거래일.
- `R_4M`은 baseline 호환용일 수 있으나 원점 후보 목록에서 제외.

### 4.2 해상도·비중첩 (필수)

| 구분 | 예시 |
|------|------|
| 누적(고해상도) | 최근 3일, 5일, 10일, 21일, 63일 |
| 비중첩 | 0–5일, 5–21일, 21–63일 (및 더 잘게 0–3, 3–5, 5–10, 10–21) |
| 구간 통계 | 구간 수익률·로그기울기·변동성, `PRIOR_SUPPORT_SIGN` |

누적만 쓰면 V자 반등과 지속 상승을 구분하기 어렵다. **비중첩을 반드시 함께 쓴다.**

---

## 5. 3단계 — 자연어 규칙 도출

```bash
python -m calibration.fms_recalib_inspect_patterns
```

1. TOP/MID/BOT 및 상위·하위 20의 피처 평균을 본다.
2. 개별 차트로 대표 성공/실패 경로를 확인한다.
3. 수식 전에 **자연어 규칙**을 `fms_recalib_natural_language_rules.json`에 남긴다.

자연어 예시:

> 장기적으로 일정한 상승에 있으면서 최근 상승이 가파르면 우수하다.  
> 다만 절대 최근 1개월 수익률이 약하면 장기 꾸준함보다 단기 상승 가중치를 키운다.  
> 1주·1개월·3개월이 모두 매우 낮으면(완전 지속 하락 제외 기준은 데이터로 조정)
> 최하권에 가깝게 둔다. 자산군 예외는 쓰지 않는다.

---

## 6. 4단계 — 비선형 수식 + 몬테카를로 경쟁

```bash
python fms_recalib_nonlinear_mc.py
```

### 6.1 수식 family

- 각 family는 자연어 의도를 연속 반응 함수로 구현한다.
- 예: softplus 바닥 게이트, regime switch, 구간 곱셈 확인, sqrt/log 혼합.
- 블랙박스 tree/boosting·종목 예외 금지.

### 6.2 몬테카를로

- family별 파라미터를 로그-균등/균등으로 대량 샘플링한다.
- development pairwise 지표(utility ≈ −inversion + Spearman − pair-delta)로 선정 후
  최적 주변 local jitter로 정제한다.
- 여러 family를 **서로 경쟁**시켜 최종 후보 1개를 고른다.
- nested holdout·label variants로 안정성을 본다. audit은 동결 후 1회.

### 6.3 레거시 sparse/GAM 경로

필요 시 `python fms_recalib_refit.py`로 benchmark 비교 가능. 승격 기본안이 아님.

---

## 7. 5단계 — 과적합 검증

- nested symbol holdout, label variants (`2^k`, k≤5), LOO/residual 차트
- audit 지표가 production보다 악화되면 승격하지 않는다
- audit을 본 뒤 수식을 바꾸면 같은 audit을 “미관측 검증 완료”로 부르지 않는다

---

## 8. 평가 지표와 채택

- `inversion_rate` ↓, `spearman_rho` ↑, `pair_delta_error` ↓, top-quintile recall
- full·development에서 production 대비 개선 + nested 안정 + 자연어 설명 가능
- 한 정답셋 개선 ≠ 미래 일반화 증명

---

## 9. 과거 승격 스냅샷 (v4.6.0, 참고용)

2026-07-29 sparse-linear 10축은 당시 정답셋 기준 승격 결과이며, v4.7.0에서
정규화만 관심종목 상대로 바뀌었다. **2026-08 재피팅은 이 수식을 출발점으로
쓰지 않는다.**

---

## 10. Monte Carlo의 역할 (갱신)

- **주 optimizer**: 비선형 family 파라미터 탐색·정제·family 간 경쟁.
- 레거시 볼록 sparse pairwise loss에 대한 “보조 MC만” 정책은 폐기한다.
- 전체 배율은 순위에 영향 없으므로 필요 시 scale 고정.
- audit 확인 후 MC로 다시 손본 후보는 같은 audit으로 재승인하지 않는다.

---

## 11. 후보 보고와 승격

### 11.1 보고 (승인 전 정지)

- 자연어 규칙과 선택 family 설명
- 수식·파라미터·정규화 정책
- production 대비 full/dev/audit 지표
- nested / label variant / 잔차 사례
- 한계(단일 정답셋, audit 사용 이력)

### 11.2 승인 후에만

1. 공유 scorer를 `core/` SSOT로 승격  
2. snapshot/feature-frame parity  
3. 앱·배치 연결, `-999` 유지  
4. pytest·harness·버전·문서 동기화  

---

## 12. 재실행 명령

```bash
# 1. 피처 + manifest
python fms_recalib_build_features.py

# 2. 패턴 요약 + 자연어 규칙 JSON
python -m calibration.fms_recalib_inspect_patterns

# 3. 비선형 MC 경쟁 적합 (기본 경로)
python fms_recalib_nonlinear_mc.py

# 4. (선택) 레거시 sparse/GAM 비교
python fms_recalib_refit.py

# 5. 잔차 차트
python -m calibration.fms_recalib_plot_residuals

# 6. 회귀
python -m pytest
python -m harness.run_fms_snapshot
```

### AI 지시 예시

> 최신 완료 세션 하나만 사용해 원점 재피팅을 수행해.  
> 먼저 패턴을 보고 자연어 규칙을 남겨. 비중첩·고해상도 피처를 쓰고,  
> 비선형 family를 몬테카를로로 경쟁시켜. production은 benchmark만.  
> 후보 보고에서 멈추고 승인 전 production을 수정하지 마.

---

## 13. 레거시 도구

`fms_recalib_evaluate_formulas.py`, `tune_vol_penalty`, `tune_weights_and_transitions`,
`fms_recalib_fit_latest.py` 등은 incremental/레거시 실험용이다. 전면 재피팅 기본
경로로 쓰지 않는다.
