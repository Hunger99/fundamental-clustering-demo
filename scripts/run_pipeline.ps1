# Runs every stage listed in the config key pipeline.stages, in order, in one run directory.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 [-Config NAME|PATH] [-RunDir DIR]
# A listed stage without src/<stage>/utils/cli.py or scripts/<stage>/run.ps1 stops the pipeline before
# a run directory is created; nothing is skipped silently.
param(
    [string]$Config,
    [string]$RunDir
)
. (Join-Path $PSScriptRoot '_common.ps1')
Initialize-Launcher

$configArgs = @()
if ($Config) { $configArgs = @('--config', $Config) }

Invoke-Python (@('-m', '00_shared.utils.cli', 'check-stages') + $configArgs)
$stages = @(Invoke-Python (@('-m', '00_shared.utils.cli', 'stages') + $configArgs) | Where-Object { $_.Trim() -ne '' })
foreach ($stage in $stages) {
    $launcher = Join-Path $PSScriptRoot "$stage\run.ps1"
    if (-not (Test-Path $launcher)) {
        throw "pipeline.stages lists '$stage' but its launcher $launcher does not exist"
    }
}

if (-not $RunDir) {
    $RunDir = @(Invoke-Python (@('-m', '00_shared.utils.cli', 'new-run') + $configArgs))[-1].Trim()
}
Write-Host "run dir: $RunDir"
Write-Host "stages: $($stages -join ', ')"

foreach ($stage in $stages) {
    Write-Host "=== $stage"
    & (Join-Path $PSScriptRoot "$stage\run.ps1") -RunDir $RunDir -Config $Config | Out-Null
}
Write-Host "pipeline finished: $RunDir"
Write-Output $RunDir
