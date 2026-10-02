# 00_shared

Infrastructure used by every stage. It contains no pipeline logic.

| Module | Responsibility |
| --- | --- |
| `code/paths.py` | Project root; external data root from `FC_DATA_DIR`, else `environment_installation/local.env`. Raises `DataRootError` if the root is missing or inside the repo. |
| `code/config.py` | Loads `configs/default.yaml` and deep-merges a variant over it (`load_config`). Also provides dotted lookup (`get`), the config name, and a settings hash. |
| `code/runs.py` | Creates run directories `outputs/archive/runs/<config>--<dataset>--<UTC %Y%m%dT%H%M%SZ>/` and writes provenance: config snapshot, packages, git HEAD + dirty flag, argv, source folder, and input sha256 per stage. `open_run` refuses to reuse a run dir under different settings. |
| `code/io.py` | Parquet read/write without index; JSON writing that handles numpy types (NaN and inf become `null`). |
| `code/plotting.py` | One matplotlib style (`apply_style`), the validated palette, `cluster_colors`, and `savefig` (PNG, dpi 150). |
| `code/logs.py` | `setup_logging(run_dir, name)`: logs to the console and to `<run>/logs/<name>.log`, with UTC timestamps. |
| `utils/cli.py` | Launcher helper: `new-run`, `stages`, `check-stages` (used by `scripts/run_pipeline.ps1`). |

Usage from any stage or experiment (with `src/` on `PYTHONPATH`):

```python
from fundclust import shared
cfg = shared.config.load_config("my_variant")      # configs/my_variant.yaml over default.yaml
run_dir = shared.runs.open_run(None, cfg, source="src")
```

Run-dir layout: `artifacts/<NN_stage>/`, `logs/`, `provenance/` (`config.yaml`, `run.json`, `inputs.json`, `stages.json`).
