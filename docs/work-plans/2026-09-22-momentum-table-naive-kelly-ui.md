# 2026-09-22 — 모멘텀 테이블·사이드바 UI 정리 (+ NAIVE_KELLY 도입/제거)

Status: `done` · product **v5.0.10** (Kelly removed)

## Pain / Intent

1. 모멘텀 테이블에 FMS와 무관한 표 전용 컬럼(`R_1W` / append `R_4M` / `R_YTD`)이 남아
   해석을 흐림. 특히 `R_YTD`는 연초 창이 짧아져도 FMS에 쓰이지 않음을 확인 후 제거.
2. (v5.0.8) 단일종목 나이브 켈리를 표/정렬에 추가했으나 **운영 체감 효과 미흡** →
   **v5.0.10에서 전면 제거**.
3. 미사용 관심종목 「재평가」 UI와 캐시 초기화 버튼 위치(도움말 아래)가 운영 불편.

## Changes (final)

### Production

- 표 전용 append 제거; 컬럼 순서 `FMS` → alive_pullback 영향도순
- 사이드바: 재평가 UI 제거; 캐시 초기화 상단; 정렬 기본 **FMS(현재)**
- Dead: `ytd_return` (v5.0.8) · **`naive_kelly` / `NAIVE_KELLY_20D` (v5.0.10)**

### Boundary

- 배치 CLI `analysis_utils.calculate_fms_for_batch` **유지**
- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import

## Harness

| Asset | What it locks |
|-------|----------------|
| `tests/unit/test_fms_scoring.py` | 골든 FMS 순위 · `R_YTD`/`R_1W`/`NAIVE_KELLY_20D` 부재 |
| Existing FMS suite | `alive_pullback` production 불변 |

```bash
python -m pytest tests/unit/test_fms_scoring.py -q
python -m pytest -q
```

## Related

- SSOT: `HARNESS_RULES.md` §0 · `.cursorrules`
