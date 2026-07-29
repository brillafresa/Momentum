# 2026-07-30 — FMS 현금성 ETF 과대평가 게이트 (v4.6.1)

Status: `promoted_to_production`

## Problem

v4.6.0 sparse-linear FMS rewarded path smoothness independently of return
magnitude. KOFR/CD-rate/money-market ETFs (e.g. 449170.KS, 459580.KS,
499660.KS) scored ~2.6 and dominated IRP discovery tops.

Root cause: approved 80-symbol fit had no cash-like assets; frozen Z-scores
treat near-perfect R² / zero drawdown / zero jumps as highly favorable while
the low `R_3M` penalty is small.

## Fix

In `core/fms_features.py`:

```text
cash_strength = low_return(R_3M) × ultra_low_vol(Vol20_Ann) × high_smooth(R2_3M)
gated_contrib = min(c, 0) + (1 - cash_strength) × max(c, 0)   # quality axes except R_3M
```

Edges (decimals): R_3M 1%→5%, Vol20_Ann 0.5%→3%, R2_3M 0.95→0.99 (`smoothstep`).

## Impact (offline)

| Panel | Result |
|-------|--------|
| Synthetic cash fixture | CASH_LIKE 2.719 → −0.584 (Δ −3.303); CASH_STAIR Δ −3.002 |
| Equity / bond-rally / noisy-low / smooth-strong | FMS unchanged (cash_strength=0) |
| Calibration 80 (`fms_20260729_154752`) | **0/80 changed** (bit-identical) |
| Spearman / inversion (80) | unchanged 0.8917 / 0.1177 |
| IRP latest scan (strength only) | 30 symbols strength>0.9 incl. the three named ETFs |

Full post-v4.6 batch price panel was not available locally; IRP scan CSV
pre-dates v4.6 and lacks 7 axes, so strength was computed from R_3M/R2/Vol
only. Score deltas for live cash ETFs use the synthetic cash path as proxy.

## Harness

- `tests/unit/test_fms_cash_like_gate.py`
- `tests/fixtures/cash_like_paths_prices_krw.csv`
- `scripts/fixtures/generate_cash_like_panel.py` (재생성기; 운영 미import)
- `harness/compare_cash_like_gate.py`

## Follow-up

- LIVE IRP/FREE 배치는 v4.7.0 상대 Z 기준으로 재실행해 배포 후 랭킹을 확인한다.
- Optional later: asset-class segmentation UI (out of scope for this fix).

Note: v4.6.1의 “80종 bit-identical” 영향은 **당시 고정 Z** 기준이다.
v4.7.0에서 normalization이 관심종목 상대 Z로 바뀌었으므로 캘리브레이션
패널의 절대 점수는 더 이상 고정되지 않는다. 현금성 게이트 정책 자체는 유지된다.
