@echo off
cd /d "%~dp0"
python tools\collect_warbands.py
if errorlevel 1 pause
