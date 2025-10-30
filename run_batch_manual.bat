@echo off
REM 스크립트가 위치한 디렉토리로 이동
cd /d "%~dp0"

echo Starting KRW Momentum Radar Batch Scan...

REM 가상환경 Python 우선 사용, 없으면 시스템 Python 사용
set "VENV_PY=venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  echo Using venv Python: %VENV_PY%
  "%VENV_PY%" run_scan_batch.py
 ) else (
  echo Using system Python
  python run_scan_batch.py
 )

echo Batch scan complete. Press any key to exit.
pause


