@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Creating .venv with Python 3.11...
    py -3.11 -m venv .venv
    if errorlevel 1 exit /b %errorlevel%
)

"%VENV_PY%" install_augur.py --profile full --upgrade-pip --prewarm-pysr --check-models --trusted-host pypi.org --trusted-host files.pythonhosted.org %*
if errorlevel 1 exit /b %errorlevel%

echo.
echo Full setup completed. Start Augur with:
echo   run_augur.bat
endlocal
