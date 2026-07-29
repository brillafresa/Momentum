# 2026-07-30 — FMS 현재 관심종목 상대 Z-score 복원 (v4.7.0)

Status: `completed`

## User intent

FMS는 고정 development 집단 대비 절대점수가 아니라 현재 계좌 관심종목을
기준으로 한 상대점수여야 한다.

- 앱: 현재 관심종목끼리 비교
- 배치: 신규 후보를 현재 계좌모드 관심종목과 비교
- `FMS=0`: 현재 관심종목 기준선

## Implementation

- `core/fms_features.py`
  - 축별 reference median 결측 대체
  - `(target - reference mean) / reference std`, Z ±4 clip
  - reference 유효값 <2 또는 std≤1e-12이면 해당 축 기여 0
  - 실격 관심종목은 reference 통계에서 제외
- `core/fms.py`
  - `reference_prices_krw`에서 reference feature frame 생성
  - reference 미지정 시 target self-reference
- `analysis_utils.py` / `run_scan_batch.py`
  - self-reference 재평가는 outer chunk별로 나누지 않고 전체 watchlist를 한 기준으로 사용
  - 신규 탐색은 유효 관심종목 reference가 2개 미만이면 중단
- 기존 sparse-linear 가중치, 현금성 게이트, 거래적합성 `-999` 유지

## Harness

- reference 구성 변경 시 FMS 변경
- reference 미지정 = target self-reference
- self-reference ungated 축별 평균 기여 0
- zero-variance reference 축 기여 0
- batch `FixtureAdapter` = 직접 scorer
- feature-frame scorer = snapshot scorer

## Measured impact

승인 80종 패널에서 frozen v4.6.1 → relative v4.7.0:

- 양수/음수: 43/37 → 42/38
- 평균 ΔFMS: -0.0287, 중앙 ΔFMS: -0.0298
- Δ 범위: -0.0979 … +0.1301
- 순위 변경: 22/80 (대부분 소폭)
- 사람 순위 Spearman: 0.8917 → 0.8933

합성 골든 순위와 `CRASHY=-999`는 유지됐다.

## Product semantics

관심종목 추가·삭제 또는 계좌모드 변경 시 FMS가 달라지는 것이 의도된 동작이다.
저장된 배치 점수는 실행 당시 해당 계좌의 관심종목 구성에 종속된다.

## Push-prep (2026-07-30)

- [x] cash fixture 생성기 `scripts/fixtures/generate_cash_like_panel.py`로 이관
- [x] 하네스 README / tests README / HARNESS_RULES §0 / TODO / CHANGELOG 동기화
- [x] production docstring의 frozen-normalization 잔여 문구 제거
- [x] `scan_results/.gitkeep` UTF-8 복구
- [x] pytest + import 스모크 통과
