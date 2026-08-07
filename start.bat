@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title bili_summary launcher
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ from python.org
    echo         and check "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)
echo ============================================================
echo   bili_summary - Bilibili video -^> transcript / summary
echo   First run creates .venv and installs dependencies automatically.
echo   No argument  : interactive mode (type links, empty line to start)
echo   Examples:
echo     bili_summary.py BV1GJ411x7h7 --no-summary
echo     bili_summary.py --config       (settings menu)
echo     bili_summary.py --show-config  (view current config)
echo ============================================================
python bili_summary.py %*
pause
