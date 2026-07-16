# tests/ — 자동 테스트 하네스

pytest 기반 **격리 검증** 자산. 라이브 시장 API 없이 FMS·필터·계약을 검증한다.

원칙: 루트 [`HARNESS_RULES.md`](../HARNESS_RULES.md) · 진행 상태: [`TODO.md`](../TODO.md)

## 구조

```
tests/
├── conftest.py          # fixture 로더 (CSV / golden JSON)
├── fixtures/            # 체크인 Mock 데이터 (버전 관리)
│   ├── synthetic_prices_krw.csv
│   ├── synthetic_ohlc.csv
│   └── golden_fms_ranks.json
├── unit/                # 순수 로직 단위 테스트
│   └── test_fms_scoring.py
└── contract/            # 아키텍처 계약 (예: core 네트워크 금지)
    └── test_no_network_in_core.py
```

## 실행

```bash
# 저장소 루트에서
python -m pytest
python -m pytest tests/unit/test_fms_scoring.py -q
```

## 규칙

- 새 비즈니스 로직은 **여기 테스트가 먼저** 존재하거나 동시에 추가되어야 한다.
- fixture는 재현 가능해야 하며, 의도된 공식 변경이 아니면 golden을 함부로 바꾸지 않는다.
- 네트워크가 필요한 검사는 단위 테스트에 넣지 않는다 (별도 smoke / scripts).
