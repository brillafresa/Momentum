# harness/ — 수동 시나리오 러너

pytest 밖에서 fixture·LIVE 데이터를 주입해 FMS를 눈으로 확인·디버그할 때 사용한다.
`app.py` / `run_scan_batch.py`는 이 디렉터리를 import하지 않는다.

## Offline (체크인 fixture)

```bash
python -m harness.run_fms_snapshot
python -m harness.run_fms_snapshot --no-ohlc
python -m harness.run_fms_snapshot --prices tests/fixtures/synthetic_prices_krw.csv

# 현금성 게이트: ungated vs gated 기여·ΔFMS (기본 패널 = cash_like fixture)
python -m harness.compare_cash_like_gate
python -m harness.compare_cash_like_gate --top 6
```

## LIVE (수동만 · CI 금지)

```bash
python -m harness.diagnose_fms_outlier 381560.KS
python -m harness.check_relative_ranks
python -m harness.check_relative_ranks --symbols KMI SU PBR
```

자동 assert가 필요하면 `tests/`에 테스트를 추가한다.
상세 SSOT: [`HARNESS_RULES.md`](../HARNESS_RULES.md) §0.
