# KRW Momentum Radar - 배치 스캔 설정 가이드 (Windows 11)

이 문서는 Windows 작업 스케줄러로 매일 자동으로 배치 스캔을 실행하는 방법을 안내합니다.

## 실행할 프로그램
- `run_batch_manual.bat`의 전체 경로 (예: `E:\Users\likedy\Projects\Momentum\run_batch_manual.bat`)

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
5. [설정] 탭 (견고성)
   - "예약된 시작 시간이 지난 경우 가능한 빨리 작업 실행" 체크
   - "작업이 실패하는 경우 다시 시작": 30분 간격, 3회
   - "작업이 이미 실행 중인 경우 다음 규칙 적용": 새 인스턴스 시작 안 함 (권장)

## 수동 실행
- 프로젝트 루트에서 `run_batch_manual.bat`를 더블 클릭
- 또는 PowerShell/명령 프롬프트에서 실행:

```bat
cd /d E:\Users\likedy\Projects\Momentum
run_batch_manual.bat
```

## 출력/결과
- 타임스탬프가 있는 결과 파일: `scan_results_YYYYMMDD_HHMMSS.csv` (프로젝트 루트 및 `scan_results/`에도 복사)
- 최신 결과 포인터: `scan_results/latest_scan_results.csv`

## 문제 해결
- Python 경로 문제: 가상환경 사용 시 `run_batch_manual.bat`에서 `call venv\Scripts\activate.bat` 주석 해제
- 네트워크/Rate limit: 자동 재시도는 스크립트에서 처리하지 않으므로, 스케줄 재시도 옵션을 활성화하세요
- 권한 문제: 관리자 권한으로 작업 스케줄러 실행


