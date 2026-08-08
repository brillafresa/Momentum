# KRW Momentum Radar - 배치 스캔 설정 가이드 (Windows 11)

이 문서는 Windows 작업 스케줄러로 매일 자동으로 배치 스캔을 실행하는 방법을 안내합니다. (Python 3.11 권장)

## 실행할 프로그램
- 프로그램: `run_batch_manual.bat`의 전체 경로 (예: `E:\Users\likedy\Projects\Momentum\run_batch_manual.bat`)
- 인수 추가: `--no-pause` (작업 스케줄러 실행 시 필수)

## 권장 스케줄
- 트리거: 매일, 미국 정규장 마감 후 (예: 오전 6:00 KST)

## 작업 스케줄러 설정 단계
1. 작업 스케줄러 열기 → 작업 만들기
2. [일반] 탭
   - 이름: KRW Momentum Radar Batch Scan
   - 보안 옵션: "사용자가 로그온했는지 여부에 관계없이 실행"
   - "가장 높은 수준의 권한으로 실행" 체크
3. [트리거] 탭
   - 새로 만들기 → 매일 → 시작 시간: 06:00:00
4. [동작] 탭
   - 새로 만들기 → 프로그램/스크립트: `run_batch_manual.bat` 전체 경로
   - 인수 추가 (선택적): `--no-pause` 입력
   - 참고: `--no-pause` 인수를 추가하면 작업 완료 후 자동으로 종료됩니다.
5. [설정] 탭 (견고성)
   - "예약된 시작 시간이 지난 경우 가능한 빨리 작업 실행" 체크
   - "작업이 실패하는 경우 다시 시작": 30분 간격, 3회
   - "작업이 이미 실행 중인 경우 다음 규칙 적용": 새 인스턴스 시작 안 함 (권장)

## 수동 실행
- **더블 클릭**: 프로젝트 루트에서 `run_batch_manual.bat`를 더블 클릭 (종료 시 키 입력 대기)
- **PowerShell/명령 프롬프트**: 
  ```bat
  cd /d E:\Users\likedy\Projects\Momentum
  run_batch_manual.bat
  ```
- **즉시 종료**: 수동 실행 시에도 키 입력을 원하지 않으면 `--no-pause` 인수 추가
  ```bat
  run_batch_manual.bat --no-pause
  ```

참고: `run_batch_manual.bat`은 자동으로 `venv\Scripts\python.exe`가 있으면 이를 사용하고, 없으면 시스템 `python`으로 실행합니다.

## 출력/결과
- 타임스탬프가 있는 결과 파일: `scan_results/scan_results_{mode}_YYYYMMDD_HHMMSS.csv`
- 최신 결과 포인터: `scan_results/latest_scan_results_{mode}.csv` (mode: free 또는 irp)
- 모든 스캔 결과 파일은 `scan_results/` 디렉토리에만 저장됩니다
- **시장 데이터 디스크 캐시 (v5.0.3+):** 다운로드 성공 청크는 `cache/market_data/`에 write-through
  (gitignore). UI·배치는 last-bar probe로 HIT/MISS 후 재사용.
  **v5.0.4:** probe는 **이미 캐시된(warm) 심볼만** — cold는 full download만 (USA 선행 청크
  레이트 실패 방지). 강제 갱신은 앱의「데이터 캐시 초기화」.
  LIVE 회귀: `python -m harness.smoke_multi_market_batch` /
  `python -m harness.smoke_usa_first_batch`.

## 동작 참고
- 배치 실행 시 유니버스를 강제로 재스크린합니다 (FREE: Finviz `set_filter` + 로컬 후처리 + `korean_universe.csv` + `hongkong_universe.csv` 병합).
- Finviz Overview 페이지 fetch는 `finviz_screener_view_resilient()`로 per-page 재시도·partial fallback 합니다 (마지막 페이지 hang 방지).
- 개발 시 Finviz 갱신 생략: `python run_scan_batch.py --skip-universe-update`
- Finviz Overview가 티커 첫 글자를 중복하는 경우(예: `AAPL`→`AAAPL`) 자동 보정합니다.
- 앱과 동일한 FMS/거래 적합성 필터 로직(`analysis_utils.py`)을 사용합니다.
- **v5.0.0 절대 FMS** (`alive_pullback`): 관심종목 reference 패널은 점수에 **영향 없음**.
  관심종목은 스캔 대상 제외·거래적합성 걸러내기에만 쓰이며, 비어 있어도 스캔을 중단하지 않습니다.
  `DEFAULT_FMS_THRESHOLD=0.0`은 저장 하한(절대 점수 ≥ 0)입니다.
- **가격은 Adj Close(배당 조정) 기준**입니다. OHLC 거래적합성 필터는 원시 High/Low/Close를 사용합니다.
- yfinance 레이트리밋(`Too Many Requests` / `shared._ERRORS`)은 지수 백오프(최대 ~40회, 대기 상한 120초)로 재시도합니다.
- 청크 간 대기·outer batch로 전체 유니버스 커버리지를 우선합니다(속도보다 완료).
- **`no data` / `possibly delisted` 메시지는 정상 스킵**입니다. 해당 심볼만 제외하고 배치는 계속됩니다.

## 문제 해결
- **프로세스 종료 안 됨**: 작업 스케줄러 실행 시 `--no-pause` 인수를 추가하지 않아서 발생하는 문제입니다. [동작] 탭의 "인수 추가"에 `--no-pause`를 입력하세요.
- Python 경로 문제: 가상환경 사용 시 `run_batch_manual.bat`에서 `call venv\Scripts\activate.bat` 주석 해제
- **UnicodeEncodeError (cp949)**: `run_batch_manual.bat`이 UTF-8(`PYTHONUTF8=1`)을 설정합니다. 직접 `python` 실행 시에도 동일 환경변수를 권장합니다.
- 네트워크/Rate limit: 스크립트가 재시도합니다. 그래도 실패하면 스케줄러 재시도(30분×3회)를 유지하세요.
- 권한 문제: 관리자 권한으로 작업 스케줄러 실행
- 로그: `scan_results/batch_{free|irp}_YYYY-MM-DD.log` (수동 실행 시 리다이렉트 권장)


