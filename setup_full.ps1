Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating .venv with Python 3.11..."
    py -3.11 -m venv .venv
}

& $venvPython install_augur.py `
    --profile full `
    --upgrade-pip `
    --prewarm-pysr `
    --check-models `
    --trusted-host pypi.org `
    --trusted-host files.pythonhosted.org `
    @args

Write-Host ""
Write-Host "Full setup completed. Start Augur with:"
Write-Host "  .\run_augur.ps1"
