# 2026-08-08 — 세부보기 캐시 + 공유 디스크 last-bar probe (v5.0.3)

Status: `done` · product **v5.0.3**

## Pain

- 세부보기 종목 전환마다 전체 rerun + FMS 3스냅샷 재계산으로 체감 지연
- 라벨/index와 차트 시계열 분리 시 잘못된 매수 판단 위험
- 배치가 받은 가격을 UI가 재사용하지 못함 (6h TTL Streamlit 메모리만)

## Fix

### Phase 1
- `adapters/ui_data_bundle.py`: panel fingerprint, `DetailViewAtom`, reconcile
- `app.py`: 세션 번들로 FMS 메모; select 고정 key + 심볼 SSOT; atom으로만 차트/배지 렌더
- 불일치 시 fail-closed 에러 배너

### Phase 2
- `adapters/price_cache.py`: `needs_refresh` / `DiskPriceCache` / `CachingMarketDataAdapter`
- 배치 `calculate_fms_for_batch` + watchlist OHLC write-through
- UI `download_*` → 캐시 어댑터 ( `@st.cache_data` 유지 )
- 캐시 초기화: Streamlit + 세션 + 디스크

## Harness

- `tests/unit/test_ui_panel_fingerprint.py`
- `tests/unit/test_detail_view_atom.py`
- `tests/unit/test_price_cache_freshness.py`

## Verification

- `python -m pytest` EXIT=0
- 수동: 세부보기 A→B→A, 추가/삭제 후 헤더 티커 = 차트 hover; 캐시 초기화 동작
