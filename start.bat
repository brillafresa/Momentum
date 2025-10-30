@echo off
REM 스크립트가 위치한 디렉토리로 이동
cd /d "%~dp0"

echo Stopping existing Python processes (if any)...
taskkill /f /im python.exe >nul 2>&1

echo Ensuring virtual environment...
if not exist "venv\Scripts\python.exe" (
  python -m venv venv
)

set "PYEXE=python"
if exist "venv\Scripts\python.exe" (
  set "PYEXE=venv\Scripts\python.exe"
  echo Using venv Python: %PYEXE%
) else (
  echo Using system Python
)

echo Installing/updating dependencies...
"%PYEXE%" -m pip install --upgrade pip --disable-pip-version-check >nul 2>&1
"%PYEXE%" -m pip install -r requirements.txt --disable-pip-version-check

echo Starting Streamlit on port 8501...
"%PYEXE%" -m streamlit run app.py --server.port 8501
