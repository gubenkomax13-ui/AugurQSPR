@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Virtual environment not found. Run setup_full.bat first.
    exit /b 1
)

"%VENV_PY%" -m streamlit run qspr_app.py
endlocal
