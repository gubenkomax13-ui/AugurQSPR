Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot "setup_full.ps1") @args
