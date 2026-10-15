@echo off
setlocal EnableExtensions
rem Checks the compiled core of the fast engine on this machine.
set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.adaptive_ground_env\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Run Run_AdaptiveGround_Unified.bat once first: it installs the Python packages this needs.
    pause
    exit /b 1
)
"%VENV_PYTHON%" "%SCRIPT_DIR%test_fast_engine.py" "%SCRIPT_DIR%validation_suite\clips"
pause
