# scripts/ — 보조 스크립트 & fixture 도구

일회성 변환, fixture 생성, 오프라인 실험용 스크립트를 둔다.  
**프로덕션 엔트리포인트(`app.py`, `run_scan_batch.py`)에 실험 코드를 넣지 않는다.**

- 반복 가능한 자동 검증 → `tests/`
- 수동 시나리오·테이블 출력 → `harness/`
- 생성기·마이그레이션·임시 분석 → **여기 `scripts/`**

## 구조

```
scripts/
├── README.md
├── analyze_prefilter_impact.py          # LIVE Finviz 사전필터 실측 (수동)
├── build_hk_universe_from_indices.py    # LIVE HK 유니버스 재생성 (운영 미import)
└── fixtures/
    ├── README.md
    ├── generate_synthetic_panel.py
    └── prefilter_band_sample_fms.csv
```

## 규칙

1. 스크립트는 기본적으로 **로컬 파일만** 읽고 쓴다. 라이브 API 호출 시 파일 상단과 CLI 플래그로 명시한다.
2. 생성된 데이터가 회귀 테스트에 쓰이면 `tests/fixtures/`로 **승격·커밋**한다.
3. FMS 점수가 필요하면 `compute_fms_snapshot` / `momentum_now_and_delta`만 호출한다 (공식 복제 금지).

## 스크립트 목록

| 파일 | 목적 | 실행 |
|------|------|------|
| `build_hk_universe_from_indices.py` | HSI/HSCEI/HSTECH → `hongkong_universe.csv` 재생성 | `python scripts/build_hk_universe_from_indices.py` |
| `analyze_prefilter_impact.py` | Finviz 사전필터 tightness LIVE 실측 | `python scripts/analyze_prefilter_impact.py` |
| `fixtures/generate_synthetic_panel.py` | `tests/fixtures/` 합성 패널 재생성 | 필요 시만 |
