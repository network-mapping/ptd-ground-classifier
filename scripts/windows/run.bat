@echo off
rem Usage: run.bat <files or folders> [--output-dir DIR] [--units usft^|metres] ...
setlocal
cd /d "%~dp0\..\.."
if not exist .venv\Scripts\ground-classify.exe (
    echo Run scripts\windows\setup.bat first.
    exit /b 1
)
.venv\Scripts\ground-classify.exe %*
