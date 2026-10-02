# Stage 02: build the configured feature set from artifacts/01_load_data/panel.parquet.
# -RunDir must point at a run that already holds stage 01 output.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Config
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
Invoke-Python (Get-StageArguments -Module '02_features.utils.cli' -RunDir $RunDir -Config $Config)
