# Stage 04: PCA, parallel analysis, PC profiles, embeddings and embedding quality into artifacts/04_reduce/.
# -RunDir must point at a run that already holds artifacts/03_denoise/X.parquet, unless -InputFile names a matrix.
# -Methods takes a comma list (umap,tsne,isomap,mds) or none for PCA only.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Config,
    [string]$InputFile,
    [string]$Denoiser,
    [string]$Methods,
    [switch]$NoFigures
)
. (Join-Path $PSScriptRoot '..\_common.ps1')
Initialize-Launcher
$cliArgs = Get-StageArguments -Module '04_reduce.utils.cli' -RunDir $RunDir -Config $Config
if ($InputFile) { $cliArgs += @('--input', $InputFile) }
if ($Denoiser) { $cliArgs += @('--denoiser', $Denoiser) }
if ($Methods) { $cliArgs += @('--methods', $Methods) }
if ($NoFigures) { $cliArgs += '--no-figures' }
Invoke-Python $cliArgs
