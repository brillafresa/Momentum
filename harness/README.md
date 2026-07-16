# harness/ — 수동 시나리오 러너

pytest 밖에서 fixture를 주입해 FMS 테이블을 눈으로 확인·디버그할 때 사용한다.

```bash
python -m harness.run_fms_snapshot
python -m harness.run_fms_snapshot --no-ohlc
python -m harness.run_fms_snapshot --prices tests/fixtures/synthetic_prices_krw.csv
```

자동 assert가 필요하면 `tests/`에 테스트를 추가한다.
