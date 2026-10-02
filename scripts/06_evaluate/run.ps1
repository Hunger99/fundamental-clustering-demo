# Stage 06: evaluate the run's clustering into artifacts/06_evaluate/.
# -RunDir must point at a run that already holds artifacts/05_cluster/{labels,space}.parquet.
# Sector labels come from the config key evaluate.external.file (relative to FC_DATA_DIR); -External overrides it.
# The optional calibration and pseudoreplication blocks run when evaluate.extra_blocks lists them or -Blocks names them.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Config,
    [string]$Blocks,
    [string]$External,
    [switch]$NoRefit
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
$cliArgs = Get-StageArguments -Module '06_evaluate.utils.cli' -RunDir $RunDir -Config $Config
if ($Blocks) { $cliArgs += @('--blocks', $Blocks) }
if ($External) { $cliArgs += @('--external', $External) }
if ($NoRefit) { $cliArgs += '--no-refit' }
Invoke-Python $cliArgs
