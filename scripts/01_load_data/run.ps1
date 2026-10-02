# Stage 01: load fund.csv, apply the missing-data policy, write artifacts/01_load_data/.
# Without -RunDir a new run directory is created under outputs/archive/runs/.
param(
    [string]$RunDir,
    [string]$Config
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
Invoke-Python (Get-StageArguments -Module '01_load_data.utils.cli' -RunDir $RunDir -Config $Config)
