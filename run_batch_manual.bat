@echo off
REM 스크립트가 위치한 디렉토리로 이동
cd /d "%~dp0"

echo Starting KRW Momentum Radar Batch Scan...

REM 가상환경이 있다면 활성화 (예: venv\Scripts\activate.bat)
REM call venv\Scripts\activate.bat

REM Python 스크립트 실행
python run_scan_batch.py

echo Batch scan complete. Press any key to exit.
pause


