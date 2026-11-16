# 2026-11-16 — 거래적합성 하방리스크: SMA5 → MED3

Status: `done` · product **v5.0.15**

## Pain / Intent

세부보기 차트 MED3(v5.0.14)와 동일하게, 거래 적합성 **반복적 하방리스크** 기준도
종가 5일 평균(SMA5) 대신 **3일 종가 중앙값(MED3)** 을 쓴다.
alive_pullback FMS 수식·치명적 변동성(TR)은 변경하지 않는다.

## Changes (final)

### Production

- `core/indicators.rolling_close_median` — MED3 SSOT
- `core/tradeability.py`: `ma5=rolling(5).mean()` → `med3=rolling_close_median(close, 3)`
  - `(low / med3) - 1 < -7%` · ≥4일/20d · shift 없음
  - Filter_Status: `MED3대비 -7% 미만`
  - debug severe detail: `med3` (`sma5` 제거)
- `adapters/ui_data_bundle.rolling_close_median` — core re-export (차트)
- `analysis_utils` — `rolling_close_median` re-export
- `app.py` 도움말·디버그 헤더 문구 동기화

### Boundary

- production FMS (`alive_pullback`) **불변**
- TR 치명적 변동성 **불변**
- Finviz `SMA50`/`SMA200` 사전필터 **무관**
- `config.py`에 fixture/Mock 경로 없음
- `app.py` / `run_scan_batch.py`는 `tests/` · `harness/` 미import
- MED3는 FMS 점수 입력이 아님 (실격 게이트 + 차트 오버레이만)

### Dead code

- 거래적합성 경로의 `ma5` / `sma5` / `SMA5대비` 문자열 제거
- adapters 로컬 `rolling_close_median` 본문 삭제 → core SSOT re-export만 유지

## FMS / 퀀트 로직 하네스 (이번 세션)

새 alive_pullback 공식은 만들지 않았다. **하방리스크 게이트만** MED3로 바꾸고,
기존 FMS·필터 하네스로 회귀를 잠근다.

| 자산 | 검증 내용 | 역할 |
|------|-----------|------|
| `tests/unit/test_tradeability.py` | MED3 −10%×4 → DQ · −6%×4 → pass · CRASHY · shim | **계약 갱신** |
| `tests/unit/test_tradeability_debug_info.py` | counts ≡ filter · `med3` 필드 · `sma5` 부재 | **계약 갱신** |
| `tests/unit/test_fms_scoring.py` | 골든 순위 · CRASHY→`-999` · reference 불변 | FMS 회귀 |
| `tests/unit/test_fms_alive_pullback_production.py` | 동결 파라미터 · disqualification `-999` | FMS 회귀 |
| `tests/unit/test_indicators.py` | `rolling_close_median` ≡ `rolling(3).median()` | MED3 SSOT |
| `tests/unit/test_detail_chart_med3.py` | median≠SMA5 mean · core≡adapters | 차트·경계 |
| `tests/contract/test_no_network_in_core.py` | `core/` 네트워크 import 금지 | 계약 |

```bash
python -m pytest tests/unit/test_tradeability.py tests/unit/test_tradeability_debug_info.py \
  tests/unit/test_fms_scoring.py tests/unit/test_detail_chart_med3.py \
  tests/unit/test_indicators.py -q
```

푸시 전 전체: `python -m pytest -q`
