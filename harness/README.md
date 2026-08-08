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

# 배치 vs UI 캘린더 경로 dFMS (동일 fixture)
python -m harness.compare_batch_ui_fms --offline
```

## LIVE (수동만 · CI 금지)

```bash
python -m harness.diagnose_fms_outlier 381560.KS
python -m harness.check_relative_ranks
python -m harness.check_relative_ranks --symbols KMI SU PBR

# 동일 다운로드 → UI(coverage=0.5) vs 배치(coverage=0.9) FMS 차이 실측
python -m harness.compare_batch_ui_fms --live
python -m harness.compare_batch_ui_fms --live --limit 40 --top 20
# UI식 시장별 다운로드 vs 배치 일괄 다운로드 (연속 실행)
python -m harness.compare_batch_ui_fms --live --mirror-io --limit 40
```

2026-08-05 실측: 연속 실행 시 max|dFMS|≈0.01·순위상관 1.0 → 경로 통일 보류.
상세: `docs/work-plans/2026-08-05-batch-ui-fms-path-compare.md`.

2026-08-07 (v5.0.2): 다국가 trailing ffill 오염은 `core.indicators.harmonize_calendar`
native as-of clip으로 수정. 오프라인 회귀:
`python -m pytest tests/unit/test_native_asof_calendar.py -q`
상세: `docs/work-plans/2026-08-07-native-asof-fms.md`.

2026-08-08 (v5.0.3): 세부보기 DetailViewAtom + 세션 FMS 메모 + 디스크 last-bar probe.
오프라인: `python -m pytest tests/unit/test_detail_view_atom.py tests/unit/test_price_cache_freshness.py -q`
상세: `docs/work-plans/2026-08-08-detail-view-cache.md`.

자동 assert가 필요하면 `tests/`에 테스트를 추가한다.
상세 SSOT: [`HARNESS_RULES.md`](../HARNESS_RULES.md) §0.
