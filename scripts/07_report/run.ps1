# Stage 07: report figures, summary.md and report.json into artifacts/07_report/.
# -RunDir must point at a run that already holds the artifacts of stages 01 to 06.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Config
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
Invoke-Python (Get-StageArguments -Module '07_report.utils.cli' -RunDir $RunDir -Config $Config)
