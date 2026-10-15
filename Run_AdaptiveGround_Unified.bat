@echo off
setlocal EnableExtensions
title Adaptive Ground Unified

set "SCRIPT_DIR=%~dp0"
set "PYTHON_CMD="

where py.exe >nul 2>&1 && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python.exe >nul 2>&1 && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo ERROR: Python 3 was not found.
    pause
    exit /b 1
)

set "VENV_DIR=%SCRIPT_DIR%.adaptive_ground_env"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 goto failed
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
"%VENV_PYTHON%" -c "import laspy, lazrs, numpy, scipy" >nul 2>&1
if errorlevel 1 (
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto failed
    "%VENV_PYTHON%" -m pip install "laspy[lazrs]" numpy scipy
    if errorlevel 1 goto failed
)

"%VENV_PYTHON%" "%SCRIPT_DIR%adaptive_ground_classifier_unified.py" %*

if errorlevel 1 goto failed
echo.
echo Adaptive Ground Unified completed successfully.
pause
exit /b 0

:failed
echo.
echo Classification failed. Review the error above.
pause
exit /b 1
