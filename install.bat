@echo off
setlocal
cd /d "%~dp0"
py install_augur.py --profile local %*
endlocal
