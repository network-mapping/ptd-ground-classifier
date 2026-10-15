@echo off
setlocal EnableExtensions
rem Compare two output folders (and optionally a reference such as TerraScan).
rem   Compare_Outputs.bat  folderA  folderB  [referenceFolder]
set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.adaptive_ground_env\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Run Run_AdaptiveGround_Unified.bat once first: it installs the Python packages this needs.
    pause
    exit /b 1
)
"%VENV_PYTHON%" "%SCRIPT_DIR%compare_ground_outputs.py" %*
pause
