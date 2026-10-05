# 2026-10-05 — 반복 하방리스크: prev_close → SMA20(전일까지)

Status: `superseded` · product **v5.0.11** (same-day follow-up: **v5.0.12 SMA5**)
Current SSOT: [`2026-10-05-tradeability-sma5-downside.md`](2026-10-05-tradeability-sma5-downside.md)

## Pain / Intent

반복적 하방리스크 실격이 **전일 종가 대비 -7%** 를 쓰면, 단기 갭·노이즈에
과도하게 민감하고 “추세 평균 대비 저가 이탈”과 의미가 어긋남.
운영 의도는 **전일까지의 종가 20일 단순이동평균** 대비 저가가 -7% 미만인 날이
최근 20거래일 중 4일 이상일 때 거래부적합(FMS=-999)으로 처리하는 것.

## Changes (final)

### Production (`core/tradeability.py`)

- `daily_downside_risk = (low_fixed / close.rolling(20).mean().shift(1)) - 1`
- Filter_Status: `반복적 하방리스크 (N일 SMA20대비 -7% 미만)`
- `get_filter_debug_info` severe detail: `sma20_prev` 표시 (구 `prev_close` 기준 제거)
- 불변: 치명적 변동성(63d TR/prev_close > 30%), OHLC 부족, 63일 미만, H=L=0 glitch

### Boundary

- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- Mock OHLC·경계 케이스는 `tests/unit/test_tradeability*.py` 에만 존재
- `config.py`에 fixture/mock 경로 없음

## Harness (what we locked / how)

| Asset | What it locks |
|-------|----------------|
| `tests/unit/test_tradeability.py` | SMA20_prev × ≥4/20d DQ · prev_close-only 급상승 경계(미실격) · CRASHY · shim |
| `tests/unit/test_tradeability_debug_info.py` | debug count ≡ filter · severe detail `sma20_prev` |
| `tests/unit/test_fms_scoring.py` | 골든 순위 · CRASHY → FMS=-999 (오케스트레이션 경로) |

```bash
python -m pytest tests/unit/test_tradeability.py tests/unit/test_tradeability_debug_info.py tests/unit/test_fms_scoring.py -q
python -m pytest -q
```

## Related

- SSOT: `HARNESS_RULES.md` §0 (v5.0.11) · `.cursorrules` 거래적합성 하방 한 줄
- CHANGELOG / TODO / README 버전 **v5.0.11**
