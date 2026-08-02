# tests/ — 자동 테스트 하네스

pytest 기반 **격리 검증** 자산. 라이브 시장 API 없이 FMS·배치 I/O 헬퍼·계약을 검증한다.

원칙: 루트 [`HARNESS_RULES.md`](../HARNESS_RULES.md) · 진행 상태: [`TODO.md`](../TODO.md)

## 구조

```
tests/
├── conftest.py          # fixture 로더 (CSV / golden JSON)
├── fixtures/            # 체크인 Mock 데이터 (버전 관리)
│   ├── synthetic_prices_krw.csv
│   ├── synthetic_ohlc.csv
│   ├── golden_fms_ranks.json
│   └── cash_like_paths_prices_krw.csv
├── unit/                # 순수 로직·헬퍼 단위 테스트
│   ├── test_fms_scoring.py          # 골든 순위 · reference 불변(v5 절대)
│   ├── test_fms_alive_pullback_production.py  # v5 동결 파라미터 · parity
│   ├── test_nonlinear_mc_features.py         # SEG_* · residual features
│   ├── test_fms_cash_like_gate.py   # legacy sparse+cash gate
│   ├── test_fms_recalib_parity.py
│   ├── test_fms_features.py
│   ├── test_market_data_port.py     # FixtureAdapter 배치 = 직접 scorer
│   └── …
└── contract/            # 아키텍처 계약 (예: core 네트워크 금지)
```

Fixture **재생성기**는 `scripts/fixtures/`에 둔다 (운영 코드 미import):

- `python scripts/fixtures/generate_synthetic_panel.py`
- `python scripts/fixtures/generate_cash_like_panel.py`

## 실행

```bash
# 저장소 루트에서
python -m pytest
python -m pytest tests/unit/ -q
python -m harness.run_fms_snapshot
python -m harness.compare_cash_like_gate
```

## 규칙

- 새 비즈니스 로직은 **여기 테스트가 먼저** 존재하거나 동시에 추가되어야 한다.
- fixture는 재현 가능해야 하며, 의도된 공식 변경이 아니면 golden을 함부로 바꾸지 않는다.
- 네트워크가 필요한 검사는 단위 테스트에 넣지 않는다 (별도 smoke / 운영 배치).
- `app.py` / `run_scan_batch.py`는 이 디렉터리를 import하지 않는다.
- 재보정·FMS 하네스: `test_calibration_session.py`(saved_at 선택),
  `test_fms_features.py`(visible-window 피처),
  `test_fms_scoring.py`(골든 순위 · reference 불변),
  `test_fms_alive_pullback_production.py`(v5 SSOT),
  `test_nonlinear_mc_features.py`(SEG_*/잔차),
  `test_fms_recalib_parity.py`(feature≡snapshot),
  `test_fms_cash_like_gate.py`(**legacy** sparse+gate).
- legacy 수식 회귀: `test_fms_recent_continuation` / `test_fms_params` / `test_fms_vol_tune` /
  `test_short_horizon_*` (`score_legacy_fms_from_feature_frame` 경로).
