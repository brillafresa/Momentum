# 2026-09-22 — 모멘텀 테이블·사이드바 UI 정리 + NAIVE_KELLY (v5.0.8)

Status: `done` · product **v5.0.8**

## Pain / Intent

1. 모멘텀 테이블에 FMS와 무관한 표 전용 컬럼(`R_1W` / append `R_4M` / `R_YTD`)이 남아
   해석을 흐림. 특히 `R_YTD`는 연초 창이 짧아져도 FMS에 쓰이지 않음을 확인 후 제거.
2. 표에 단일종목 나이브 켈리(`mean/var` of 20d daily returns)를 추가해 정렬·해석에 사용.
3. 미사용 관심종목 「재평가」 UI와 캐시 초기화 버튼 위치(도움말 아래)가 운영 불편.

## Changes

### Production

- `core/indicators.naive_kelly` → `NAIVE_KELLY_20D` (rf=0, no covariance, native as-of)
- `momentum_now_and_delta`: 표 전용 append 제거; 켈리 부착; 기본 정렬=켈리 내림차순
- `MOMENTUM_TABLE_FMS_FEATURE_ORDER`: FMS → 켈리 → alive_pullback 영향도순 피처
- 사이드바: 재평가 UI·`get_button_states`·앱 `calculate_fms_for_batch` 래퍼 제거
- 「데이터 캐시 초기화」를 도구·도움말 expander 상단으로 이동
- Dead: `ytd_return` (소비자 없음) · YTD 기반 min-period 요구(252d) 제거

### Boundary

- 배치 CLI `analysis_utils.calculate_fms_for_batch` **유지** (smoke / `run_scan_batch`)
- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- Kelly는 FMS 입력 아님 (진단·정렬 전용)

## Harness built / how verified

| Asset | What it locks |
|-------|----------------|
| `tests/unit/test_naive_kelly.py` | mean/var 계약 · 영분산 · 단기 · native as-of |
| `tests/unit/test_fms_scoring.py` | 골든 FMS 순위(값 기준) · `NAIVE_KELLY_20D` 존재 · `R_YTD`/`R_1W` 부재 · 기본 인덱스=켈리 정렬 |
| `tests/unit/test_fms_cash_like_gate.py` | 골든 순위 assert를 FMS 값 기준으로 완화 (정렬 키 변경 호환) |
| Existing FMS suite | `alive_pullback` production 불변 (SEG_* / `-999` / reference 무시) |

Commands:

```bash
python -m pytest tests/unit/test_naive_kelly.py tests/unit/test_fms_scoring.py -q
python -m pytest -q
python -c "import app; import run_scan_batch"   # import smoke (no Streamlit run)
```

## Related

- Prior: `docs/work-plans/2026-08-13-ipo-native-span-coverage.md` (v5.0.6)
- SSOT: `HARNESS_RULES.md` §0 · `.cursorrules`
