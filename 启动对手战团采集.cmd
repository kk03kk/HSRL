@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Use the Python environment configured for this project.
  pause
  exit /b 1
)
python tools\collect_opponents.py watch %*
pause
