# scripts/fixtures/

Mock 가격·OHLC·피처 CSV를 **생성하거나 변환**하는 스크립트와, 커밋 전 검토용 산출물을 둔다.

- 회귀에 쓰는 확정 fixture → `tests/fixtures/`
- 생성 로직·임시/실측 덤프 → 여기
- **프로덕션(`app.py` / `run_scan_batch.py`)은 이 경로를 import하지 않는다.**

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_synthetic_panel.py` | seed=42 합성 KRW/OHLC를 `tests/fixtures/`에 기록 |

```bash
python scripts/fixtures/generate_synthetic_panel.py
python -m pytest tests/unit/test_fms_scoring.py -q
```

## Evidence dumps (checked in)

| File | Purpose |
|------|---------|
| `prefilter_band_sample_fms.csv` | 2026-07-17 Finviz 사전필터 경계 밴드 실측 FMS 샘플 (**Q+10%/H+20% 시대** LIVE 산출물; 현행 정책은 Q/H Up). 재생성: `python scripts/analyze_prefilter_impact.py --sample N` |
