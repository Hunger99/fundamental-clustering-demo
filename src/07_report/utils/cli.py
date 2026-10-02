"""Stage 07 command line: write the run's report figures, ``summary.md`` and ``report.json``.

    python -m 07_report.utils.cli --run-dir DIR [--config NAME|PATH]

Reads the artifacts of stages 01 to 06 in ``DIR`` (``01_load_data/panel.parquet``,
``02_features/{features.parquet,feature_meta.json}``, ``03_denoise/X.parquet``,
``04_reduce/{pca.json,pca_scores.parquet,pca_profiles.parquet}``,
``05_cluster/{labels.parquet,cluster_meta.json}``, ``06_evaluate/panel.json``) and the
sector file named by ``evaluate.external``. Writes ``artifacts/07_report/figures/*.png``, ``summary.md``
and ``report.json``. Settings come from the config's optional ``report:`` section.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fundclust import report, shared

STAGE = "07_report"
INPUTS = {
    "panel": ("01_load_data", "panel.parquet"),
    "features": ("02_features", "features.parquet"),
    "feature_meta": ("02_features", "feature_meta.json"),
    "X": ("03_denoise", "X.parquet"),
    "pca": ("04_reduce", "pca.json"),
    "pca_scores": ("04_reduce", "pca_scores.parquet"),
    "pca_profiles": ("04_reduce", "pca_profiles.parquet"),
    "labels": ("05_cluster", "labels.parquet"),
    "cluster_meta": ("05_cluster", "cluster_meta.json"),
    "panel_06": ("06_evaluate", "panel.json"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"{STAGE}.utils.cli", description=__doc__.splitlines()[0])
    # Required: a fresh run dir would hold none of the artifacts the report reads.
    parser.add_argument("--run-dir", required=True, help="run dir holding the artifacts of stages 01-06")
    parser.add_argument("--config", default=None, help="variant name under configs/ or a YAML path")
    args = parser.parse_args(argv)

    cfg = shared.config.load_config(args.config)
    started = shared.runs.utc_stamp()
    run_dir = shared.runs.open_run(args.run_dir, cfg, source="src", argv=list(sys.argv))
    log = shared.logs.setup_logging(run_dir, STAGE)
    out = shared.runs.stage_dir(run_dir, STAGE)
    art = run_dir / "artifacts"
    files = {k: art / a / b for k, (a, b) in INPUTS.items()}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        log.error("report inputs missing: %s", missing)
        raise FileNotFoundError(f"report inputs missing: {missing}")
    result = report.stage.run_report(run_dir, cfg, out, log=log)
    # Optional inputs: winsor bounds for the clipped-median marker, and the sector file when it was found.
    optional = {"denoise_meta": art / "03_denoise" / "denoise_meta.json", "sectors": Path(result["sector_source"])}
    files.update({k: p for k, p in optional.items() if p.is_file()})
    shared.runs.record_inputs(run_dir, STAGE, files)
    log.info("wrote %d figures, summary.md and report.json to %s", len(result["figures"]), out)
    shared.runs.record_stage(run_dir, STAGE, cfg, started, figures=sorted(result["figures"]))
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
