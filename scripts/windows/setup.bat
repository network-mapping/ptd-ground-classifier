@echo off
rem Creates ..\..\.venv and installs the package. Needs Python 3.10+ (py launcher or python on PATH).
setlocal
cd /d "%~dp0\..\.."
where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")
%PY% -m venv .venv || exit /b 1
.venv\Scripts\python -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python -m pip install -e . || exit /b 1
echo.
echo Done. The compiled core must exist as build\ptdcore.dll (see native\build.sh for the clang commands).
