@echo off
cd /d "%~dp0frontend"
where npm >nul 2>nul
if errorlevel 1 (
  echo Install Node.js LTS, then run this file again.
  pause
  exit /b 1
)
if not exist "node_modules" (
  call npm ci
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
echo Open http://localhost:5173 when Vite is ready. Keep this window open.
call npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
pause
