@echo off
rem Engine self-checks; ends with ALL OK.
setlocal
cd /d "%~dp0\..\.."
.venv\Scripts\python tests\test_fast_engine.py tests\validation\clips
