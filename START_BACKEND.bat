@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 with the Python launcher, then run this file again.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
if errorlevel 1 (
  pause
  exit /b 1
)
set DEMO_MODE=true
echo Starting Supplychainer API. Keep this window open.
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
