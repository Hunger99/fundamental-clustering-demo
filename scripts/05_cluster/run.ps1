# Stage 05: cluster artifacts/03_denoise/X.parquet (or the matrix named by -InputPath) into artifacts/05_cluster/.
# -RunDir must point at a run that already holds the input artifact; -Selection also writes K-selection curves.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Config,
    [string]$InputPath,
    [string]$Algorithm,
    [int]$K,
    [ValidateSet('', 'row', 'profile')][string]$Unit = '',
    [switch]$Selection
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
$cliArgs = Get-StageArguments -Module '05_cluster.utils.cli' -RunDir $RunDir -Config $Config
if ($InputPath) { $cliArgs += @('--input', $InputPath) }
if ($Algorithm) { $cliArgs += @('--algorithm', $Algorithm) }
if ($K -gt 0) { $cliArgs += @('--k', "$K") }
if ($Unit) { $cliArgs += @('--unit', $Unit) }
if ($Selection) { $cliArgs += '--selection' }
Invoke-Python $cliArgs
