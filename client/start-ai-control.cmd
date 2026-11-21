@echo off
setlocal
set "HSAI_DESKTOP_PY=%~dp0..\..\..\..\.venv\Scripts\pythonw.exe"
if not exist "%HSAI_DESKTOP_PY%" (
    echo HSAI Python environment was not found.
    pause
    exit /b 1
)
start "" "%HSAI_DESKTOP_PY%" -X utf8 "%~dp0ai_control.pyw"
