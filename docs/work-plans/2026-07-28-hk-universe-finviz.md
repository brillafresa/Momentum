# 2026-07-28 — FREE 홍콩 유니버스 + Finviz 페이지네이션 복원력 (v4.5.0)

## 목표

- FREE 모드에 홍콩 상장 종목(HSI·HSCEI·HSTECH 합집합) 추가
- HKD→KRW 환산을 미국/일본과 동일 FX 경로로 통합
- Finviz 유니버스 갱신 시 마지막 페이지(54/55) hang 회귀 제거
- 하네스·문서·버전 동기화 후 푸시 준비

## 구현 요약

| 영역 | 변경 |
|------|------|
| 유니버스 | `hongkong_universe.csv` (108종); `load_universe_file(FREE)` 3-way merge |
| classify | `.HK → HKG` (`analysis_utils`, `app.py`) |
| FX | `HKDKRW = USDKRW / HKDUSD`; `MarketDataPort.get_fx()` 4-tuple |
| Finviz | `finviz_screener_view_resilient()` — timeout, 5× backoff, partial fallback |
| 배치 CLI | `--skip-universe-update` (개발용) |

## 검증 하네스

```bash
python -m pytest tests/unit/test_hk_classify.py -q
python -m pytest tests/unit/test_hk_fx_conversion.py -q
python -m pytest tests/unit/test_hk_universe_loader.py -q
python -m pytest tests/unit/test_finviz_screener_pagination.py -q
python -m pytest tests/contract/test_prefilter_not_stricter_than_local.py -q
python -m pytest -q
```

## 운영/하네스 경계

- `app.py`, `run_scan_batch.py`: tests/harness/fixture 미import
- HK 유니버스 재생성: `scripts/build_hk_universe_from_indices.py` (LIVE, 수동)
- `config.py`: 운영 설정만; Mock/fixture 경로 없음

## 완료 기준

- [x] pytest 전체 통과
- [x] import smoke (`run_scan_batch`, `universe_utils`, `analysis_utils`)
- [x] CHANGELOG / TODO / HARNESS_RULES / `.cursorrules` / README v4.5.0 동기화
