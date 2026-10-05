# 2026-10-05 — 반복 하방리스크: SMA20_prev → SMA5(최신 가용)

Status: `done` · product **v5.0.12**

## Pain / Intent

v5.0.11의 SMA20(`shift(1)`, 전일까지)은 창이 길고 당일 종가를 빼서,
운영 의도와 어긋남. 짧은 **5일 단순이동평균**으로 바꾸고, 장중/마감 분기 없이
**패널에 있는 최신 Close까지** 포함한 SMA를 쓴다.

## Changes (final)

### Production (`core/tradeability.py`)

- `ma5 = close.rolling(5, min_periods=5).mean()` (no `shift`)
- `daily_downside_risk = (low_fixed / ma5) - 1`
- Filter_Status: `반복적 하방리스크 (N일 SMA5대비 -7% 미만)`
- debug severe detail: `sma5`
- 불변: 임계 −7% · 20일 창 · ≥4일 · 치명적 변동성 TR · OHLC/63일 게이트

### Boundary

- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- Mock은 `tests/unit/test_tradeability*.py` 에만

## Harness

| Asset | What it locks |
|-------|----------------|
| `tests/unit/test_tradeability.py` | SMA5 −10%×4 → DQ · −6%×4 → pass · CRASHY · shim |
| `tests/unit/test_tradeability_debug_info.py` | counts ≡ filter · `sma5` detail field |
| `tests/unit/test_fms_scoring.py` | CRASHY → FMS=-999 |

```bash
python -m pytest tests/unit/test_tradeability.py tests/unit/test_tradeability_debug_info.py tests/unit/test_fms_scoring.py -q
```

## Related

- SSOT: `HARNESS_RULES.md` §0 (v5.0.12) · `.cursorrules`
- Prior: `docs/work-plans/2026-10-05-tradeability-sma20-downside.md` (v5.0.11)
