# environment_installation

Installation files and machine-specific settings.

| File | Tracked | Purpose |
| --- | --- | --- |
| `requirements.txt` | yes | Direct dependencies with lower bounds. |
| `requirements.lock.txt` | yes | Exact versions tested here (`uv pip freeze`). This is what `install.ps1` installs. |
| `install.ps1` | yes | Creates `.venv/` with Python 3.11, installs the lock file, creates `local.env` from the example if missing, then runs the check. |
| `check_environment.py` | yes | PASS/FAIL per check: Python 3.11, imports and versions, `src/` importable, data root resolvable, sha256 of `fund.csv`. Exits 1 on any failure. |
| `local.env.example` | yes | Template for `local.env`. |
| `local.env` | no | `FC_DATA_DIR=<external data root>`. Machine-specific. |
| `.venv/` | no | The virtual environment. |
| `.cache/` | no | Tool caches (uv, pip, pytest). |

## Install

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File environment_installation\install.ps1
# newest versions within the lower bounds instead of the lock:
powershell -NoProfile -ExecutionPolicy Bypass -File environment_installation\install.ps1 -Unpinned
```

- With uv, `install.ps1` uses `uv venv --python 3.11`, which downloads Python 3.11 if needed. uv was 0.12.5 when tested.
- Without uv, it falls back to `py -3.11 -m venv` (or `python -m venv`) plus pip.

## Data root

The data stays outside the repository. Point `FC_DATA_DIR` at a folder containing:

- `raw/fund.csv`
- `raw/SHARADAR_INDICATORS_a3407c5c2ec46991d2b7ac667785d0d1.csv`
- `external/sectors_nasdaq_screener.csv`, the sector labels stage 06 scores against (`evaluate.external.file`)
- `external/sectors_sp500_gics.csv`, needed only for the optional S&P 500 sector check in stage 06 (`--external`)

Set it either as an environment variable, which takes precedence, or in `local.env`. On this machine it is `D:\Python Project\data\fundamental_clustering`. Put downloaded external data (e.g. sector labels) under `<root>/external/` with a provenance note, and reusable caches under `<root>/cache/`.

## Check

```powershell
environment_installation\.venv\Scripts\python.exe environment_installation\check_environment.py
```

## Updating dependencies

```powershell
uv pip install --python environment_installation\.venv\Scripts\python.exe <package>
uv pip freeze --python environment_installation\.venv\Scripts\python.exe > environment_installation\requirements.lock.txt
```

Add the package to `requirements.txt` with a lower bound as well.

## Tested environment (2026-10-01)

| Component | Version |
| --- | --- |
| Python | 3.11.16 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| umap-learn | 0.5.12 |
| numba | 0.68.0 |
| xgboost | 3.2.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| plotly | 7.1.0 |
| pyarrow | 25.0.1 |
| pytest | 9.1.1 |

Machine: Windows 11, 32 logical CPUs, 32 GB RAM. The launchers cap numba/OpenMP at 8 threads.
