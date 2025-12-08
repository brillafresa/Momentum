@echo off
REM 스크립트가 위치한 디렉토리로 이동
cd /d "%~dp0"

echo Starting KRW Momentum Radar Batch Scan...

REM 1) venv 없으면 생성 시도
if not exist "venv\Scripts\python.exe" (
  echo No venv found. Creating virtual environment...
  python -m venv venv
)

REM 2) 실행 Python 결정 (venv 우선)
if exist "venv\Scripts\python.exe" (
  set PYEXE=venv\Scripts\python.exe
  echo Using venv Python
) else (
  set PYEXE=python
  echo Using system Python
)

REM 3) 의존성 설치/업데이트
echo Ensuring dependencies are installed...
"%PYEXE%" -m pip install --upgrade pip >nul 2>&1
"%PYEXE%" -m pip install -r requirements.txt >nul 2>&1

REM 4) 배치 스캔 실행 (모드 파라미터 전달)
REM 첫 번째 인자: 모드 (FREE/IRP, 없으면 둘 다 실행)
REM 두 번째 인자: --no-pause (선택사항, 작업 스케줄러 실행 시 사용)

set NO_PAUSE=0
set BATCH_MODE=

REM 첫 번째 인자 확인
if "%~1"=="" goto run_both
if "%~1"=="--no-pause" (
  set NO_PAUSE=1
  goto run_both
)
if "%~1"=="FREE" (
  set BATCH_MODE=FREE
  goto mode_set
)
if "%~1"=="IRP" (
  set BATCH_MODE=IRP
  goto mode_set
)
set BATCH_MODE=%~1

:mode_set
REM 두 번째 인자 확인
if "%~2"=="--no-pause" set NO_PAUSE=1

REM 모드가 지정되지 않았으면 FREE와 IRP 모두 실행
if "%BATCH_MODE%"=="" goto run_both

REM 지정된 모드만 실행
echo Running batch scan in mode: %BATCH_MODE%
"%PYEXE%" run_scan_batch.py --mode %BATCH_MODE%
goto end_scan

:run_both
echo ========================================
echo Running batch scan for FREE mode...
echo ========================================
"%PYEXE%" run_scan_batch.py --mode FREE
if errorlevel 1 (
  echo [ERROR] FREE mode batch scan failed!
) else (
  echo [SUCCESS] FREE mode batch scan completed.
)

echo.
echo ========================================
echo Running batch scan for IRP mode...
echo ========================================
"%PYEXE%" run_scan_batch.py --mode IRP
if errorlevel 1 (
  echo [ERROR] IRP mode batch scan failed!
) else (
  echo [SUCCESS] IRP mode batch scan completed.
)

echo.
echo ========================================
echo All batch scans completed.
echo ========================================

:end_scan

REM 5) 파라미터로 pause 여부 결정
if %NO_PAUSE%==1 (
  echo Batch scan complete. Exiting...
  exit /b 0
) else (
  echo Batch scan complete. Press any key to exit.
  pause
)
