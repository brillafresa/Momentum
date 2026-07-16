# scripts/fixtures/

Mock 가격·OHLC·피처 CSV를 **생성하거나 변환**하는 스크립트와, 커밋 전 검토용 산출물을 둔다.

- 회귀에 쓰는 확정 fixture → `tests/fixtures/`
- 생성 로직·임시 덤프 → 여기

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_synthetic_panel.py` | seed=42 합성 KRW/OHLC를 `tests/fixtures/`에 기록 |

```bash
python scripts/fixtures/generate_synthetic_panel.py
python -m pytest tests/unit/test_fms_scoring.py -q
```
