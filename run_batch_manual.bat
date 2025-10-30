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
set "PYEXE=python"
if exist "venv\Scripts\python.exe" (
  set "PYEXE=venv\Scripts\python.exe"
  echo Using venv Python: %PYEXE%
) else (
  echo Using system Python
)

REM 3) 의존성 설치/업데이트
echo Ensuring dependencies are installed...
"%PYEXE%" -m pip install --upgrade pip --disable-pip-version-check >nul 2>&1
"%PYEXE%" -m pip install -r requirements.txt --disable-pip-version-check >nul 2>&1

REM 4) 배치 스캔 실행
"%PYEXE%" run_scan_batch.py

echo Batch scan complete. Press any key to exit.
pause


