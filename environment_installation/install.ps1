# Creates environment_installation\.venv with Python 3.11 and installs the pinned lock file.
#   powershell -NoProfile -ExecutionPolicy Bypass -File environment_installation\install.ps1 [-Unpinned]
# -Unpinned installs requirements.txt (newest versions within the lower bounds) instead of the lock.
# uv is preferred because it can fetch Python 3.11 itself; without uv, the 'py -3.11' launcher or a
# python on PATH builds the venv with the standard venv module and pip.
param([switch]$Unpinned)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$venv = Join-Path $here '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
$req = if ($Unpinned) { Join-Path $here 'requirements.txt' } else { Join-Path $here 'requirements.lock.txt' }
# Tool caches stay inside the installation directory (git-ignored), not in the user profile.
$env:UV_CACHE_DIR = Join-Path $here '.cache\uv'
$env:PIP_CACHE_DIR = Join-Path $here '.cache\pip'

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    if (-not (Test-Path $py)) { & uv venv --python 3.11 $venv; if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' } }
    & uv pip install --python $py -r $req
    if ($LASTEXITCODE -ne 0) { throw 'uv pip install failed' }
} else {
    Write-Warning 'uv not found; falling back to python -m venv + pip'
    if (-not (Test-Path $py)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) { & py -3.11 -m venv $venv } else { & python -m venv $venv }
        if ($LASTEXITCODE -ne 0) { throw 'python -m venv failed (Python 3.11 required)' }
    }
    & $py -m pip install --upgrade pip
    & $py -m pip install -r $req
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

if (-not (Test-Path (Join-Path $here 'local.env'))) {
    Copy-Item (Join-Path $here 'local.env.example') (Join-Path $here 'local.env')
    Write-Warning 'Created local.env from the example; edit FC_DATA_DIR to point at the data root.'
}
& $py (Join-Path $here 'check_environment.py')
exit $LASTEXITCODE
