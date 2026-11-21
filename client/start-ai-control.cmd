@echo off
setlocal
set "HSAI_CONTROL=%~dp0..\artifacts\local\r1-player150-live\client\start-ai-control.cmd"
if not exist "%HSAI_CONTROL%" (
    echo The local Player150 checkout is not installed.
    exit /b 1
)
call "%HSAI_CONTROL%"
