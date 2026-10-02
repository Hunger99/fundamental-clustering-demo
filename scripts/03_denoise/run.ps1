# Stage 03: denoise and scale the clustering features from artifacts/02_features/.
# -RunDir must point at a run that already holds stage 02 output.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Config
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
Invoke-Python (Get-StageArguments -Module '03_denoise.utils.cli' -RunDir $RunDir -Config $Config)
