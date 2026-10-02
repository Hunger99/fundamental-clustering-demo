# Shared launcher setup, dot-sourced by every script in scripts/.
# Resolves the project venv python, puts src/ on PYTHONPATH (the import root), loads
# environment_installation/local.env and caps native thread pools.

$ErrorActionPreference = 'Stop'
$script:ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Get-VenvPython {
    $py = Join-Path $script:ProjectRoot 'environment_installation\.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) {
        throw "venv python not found at $py; run environment_installation\install.ps1 first"
    }
    return $py
}

function Import-LocalEnv {
    $envFile = Join-Path $script:ProjectRoot 'environment_installation\local.env'
    if (-not (Test-Path $envFile)) { return }
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        $text = $line.Trim()
        if ($text -eq '' -or $text.StartsWith('#') -or -not $text.Contains('=')) { continue }
        $pair = $text.Split('=', 2)
        $key = $pair[0].Trim()
        $value = $pair[1].Trim().Trim('"').Trim("'")
        # A variable already set in the environment wins, the same order src/00_shared/code/paths.py uses.
        if (-not [Environment]::GetEnvironmentVariable($key, 'Process')) {
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
}

function Initialize-Launcher {
    Import-LocalEnv
    $env:PYTHONPATH = Join-Path $script:ProjectRoot 'src'
    $env:PYTHONIOENCODING = 'utf-8'
    # Several jobs share this 32-thread machine; 8 threads per numba/OpenMP pool unless the caller set one.
    foreach ($name in 'NUMBA_NUM_THREADS', 'OMP_NUM_THREADS') {
        if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) {
            [Environment]::SetEnvironmentVariable($name, '8', 'Process')
        }
    }
}

function Invoke-Python {
    # Runs the venv python with the given arguments; stdout is returned to the caller, stderr (logs)
    # goes to the console. A non-zero exit code becomes a terminating error so no stage is skipped silently.
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $py = Get-VenvPython
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-StageArguments {
    # Builds the CLI arguments shared by every stage launcher.
    param([string]$Module, [string]$RunDir, [string]$Config)
    $cliArgs = @('-m', $Module)
    if ($RunDir) { $cliArgs += @('--run-dir', $RunDir) }
    if ($Config) { $cliArgs += @('--config', $Config) }
    return ,$cliArgs
}
