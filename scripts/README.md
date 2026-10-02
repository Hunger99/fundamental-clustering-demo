# scripts

PowerShell launchers for the pipeline. They handle arguments, the environment and the order of stages, and call the stage code in `src/<stage>/utils/cli.py`. Logs and run state go to the run directory, never here.

## Run the pipeline

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Config <name or path>
```

`run_pipeline.ps1` runs every stage listed under `pipeline.stages` in the configuration, in order, into one new run directory `outputs/archive/runs/<config>--sharadar_sf1_2016q1-2020q3--<UTC>/` with `artifacts/`, `logs/` and `provenance/`. With the production configuration this takes about 2 minutes on 8 threads. `-Config` names a variant that is deep-merged over `configs/default.yaml`. A listed stage without its CLI or launcher stops the pipeline before a run directory is created.

## Run one stage

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\03_denoise\run.ps1 -RunDir <run directory>
```

Every stage launcher takes `-RunDir` and `-Config`. Stage 01 creates a new run directory when `-RunDir` is omitted; every later stage needs a run that already holds the outputs of the stages before it. A run refuses a configuration whose hash differs from the one in its `provenance/`. Extra options:

| Launcher | Options |
| --- | --- |
| `04_reduce/run.ps1` | `-InputFile` (another matrix), `-Denoiser`, `-Methods` (comma list of umap, tsne, isomap, mds, or none for PCA only), `-NoFigures` |
| `05_cluster/run.ps1` | `-InputPath`, `-Algorithm`, `-K`, `-Unit row` or `profile`, `-Selection` (also writes K-selection curves) |
| `06_evaluate/run.ps1` | `-Blocks`, `-External` (sector file instead of `evaluate.external.file`), `-NoRefit` |

## Environment

Each launcher dot-sources `_common.ps1`. It finds `environment_installation/.venv`, puts `src/` on `PYTHONPATH`, loads `environment_installation/local.env` (for `FC_DATA_DIR`) and caps `NUMBA_NUM_THREADS` and `OMP_NUM_THREADS` at 8 unless they are already set.

## Bank a run

A finished run stays in `outputs/archive/runs/`. After checking it (stage-06 gates in `artifacts/06_evaluate/panel.json`, figures in `artifacts/07_report/figures/`), move it by hand to `outputs/bank/clustering/` and move the previous bank run back to `outputs/archive/runs/`, so the bank holds only the latest verified production run. The README figures are the PNGs of that run, which `.gitignore` lets git track.
