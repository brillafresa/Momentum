# 2026-11-16 — 세부 보기 MED3 오버레이

Status: `done` · product **v5.0.14**

## Pain / Intent

세부 보기 Rebased(100) 차트의 `5일 MA` 추세선을 **3일 종가 중앙값**으로 바꾸고
표시명을 `MED3`으로 통일. UI 차트만; FMS·거래적합성 기준은 변경하지 않음.

## Changes (final)

### Production / UI

- `adapters/ui_data_bundle.rolling_close_median(s, window=3)` — 순수 헬퍼 (네트워크·Streamlit 없음)
- `app.py` 세부 보기: `ma5` / `5일 MA` → `med3` / `MED3`
- 스타일(연한 회색 실선) 유지

### Boundary

- `core/tradeability.py` SMA5 하방리스크 **불변**
- production FMS (`alive_pullback`) **불변**
- `config.py`에 fixture/Mock 경로 없음
- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- MED3는 차트 오버레이 전용 — 스코어 입력으로 쓰지 않음

## FMS / 퀀트 로직 하네스 (이번 세션)

새 FMS 공식을 만들지 않았다. 기존 오프라인 하네스로 **회귀만** 잠근다.

| 자산 | 검증 내용 | 역할 |
|------|-----------|------|
| `tests/unit/test_detail_chart_med3.py` | 3일 median · leading NaN · SMA5 mean과 비동일 | **신규** UI MED3 |
| `tests/unit/test_fms_scoring.py` | 골든 순위 · CRASHY→`-999` · reference 불변 | FMS 회귀 |
| `tests/unit/test_tradeability.py` | SMA5 −10%×4 실격 · −6%×4 통과 | 필터 회귀 |
| `tests/unit/test_tradeability_debug_info.py` | debug `sma5` 필드 · count ≡ filter | 필터 회귀 |
| `tests/unit/test_detail_view_atom.py` | 심볼↔시리즈 fail-closed | 세부보기 회귀 |

```bash
python -m pytest tests/unit/test_detail_chart_med3.py tests/unit/test_detail_view_atom.py \
  tests/unit/test_tradeability.py tests/unit/test_tradeability_debug_info.py \
  tests/unit/test_fms_scoring.py -q
```

푸시 전 전체: `python -m pytest -q`

## Dead code / 정리

- app 세부보기에서 `ma5` / `5일 MA` 문자열 제거 (잔존 참조 없음)
- 거래적합성 경로의 `ma5`/`sma5`는 **의도적 유지** (필터 SSOT)
