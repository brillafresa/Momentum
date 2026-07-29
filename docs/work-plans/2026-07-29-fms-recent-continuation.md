# 2026-07-29 — FMS 최근 우상향 가중 튜닝 (v4.5.1)

## 목표

- 계단식 상승 후 1M 평탄/완만 하강 패턴이, 최근 꾸준 우상향보다 과도하게 높은 FMS를 받지 않도록 조정
- vol 패널티·R_3M/R_4M 축은 유지 (과잉 재배치 금지)

## 원인 (LIVE 항 분해)

- 사용자 가설(R_3M/R_4M 잔존)은 부분적으로 맞음
- **최대 단일 왜곡**: `quality_mask` 실패(R2 hard >0.85) + R_1M>30% → `r1_bad` 이벤트 급등 감점
  - 예: 291620.KS R_1M≈+42%, R2≈0.83 → r1neg ≈ −0.79
  - 평탄 예시(DVA 등)는 R_1M≈0이라 해당 패널티 없음

## 구현

| 항목 | 내용 |
|------|------|
| `r1_bad` 면제 | `R_10D>0` ∧ `EMA20_SLOPE_10D>0` |
| R² quality | hard 0.85 → soft center **0.80** (`_r1_quality_weight`) |
| 가중 | `w_recent` +25%, `w_ema_shape` +15% |
| SSOT | `_r1_conditional_series` in feature-frame + `_mom_snapshot` |
| UI | 좌측 [도구 및 도움말] FMS 설명 — soft quality / continuation 면제 / 단기 연속 축 반영 |

## 검증

```bash
python -m pytest -q
python -m harness.run_fms_snapshot
```

LIVE post-tune (관심종목 참조, 2026-07-29): 291620.KS FMS ≈ 2.07 (r1_bad=0); DVA/CLDT/LFST/FLXS 소폭 하락.

## 완료 기준

- [x] pytest 전체 통과
- [x] 골든 순위 유지
- [x] CHANGELOG / TODO / HARNESS_RULES / README / `.cursorrules` / app.py v4.5.1
- [x] UI 사이드바 FMS 설명 동기화
- [x] 푸시 전 import 스모크 (`app`, `run_scan_batch`) · 프로덕션↔하네스 경계 확인
