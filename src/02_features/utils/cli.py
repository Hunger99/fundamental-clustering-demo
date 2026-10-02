"""Stage 02 command line: build the configured feature set from the stage 01 panel.

    python -m 02_features.utils.cli --run-dir DIR [--config NAME|PATH]

Reads ``artifacts/01_load_data/panel.parquet``; writes ``artifacts/02_features/features.parquet``,
``feature_meta.json`` and ``feature_report.json``. The data-quality flag counts are logged and kept in
``feature_meta.json`` under ``data_quality``.
"""

from __future__ import annotations

import argparse
import sys

from fundclust import features, shared

STAGE = "02_features"
PANEL = ("01_load_data", "panel.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"{STAGE}.utils.cli", description=__doc__.splitlines()[0])
    # Required: without it open_run would create a fresh run dir that has no panel and fail after making it.
    parser.add_argument("--run-dir", required=True, help="run dir holding artifacts/01_load_data/panel.parquet")
    parser.add_argument("--config", default=None, help="variant name under configs/ or a YAML path")
    args = parser.parse_args(argv)

    cfg = shared.config.load_config(args.config)
    get = shared.config.get
    started = shared.runs.utc_stamp()
    run_dir = shared.runs.open_run(args.run_dir, cfg, source="src", argv=list(sys.argv))
    log = shared.logs.setup_logging(run_dir, STAGE)
    out = shared.runs.stage_dir(run_dir, STAGE)

    panel_path = run_dir / "artifacts" / PANEL[0] / PANEL[1]
    panel = shared.io.read_parquet(panel_path)
    shared.runs.record_inputs(run_dir, STAGE, {"panel": panel_path})
    name = get(cfg, "features.set")
    df, meta = features.registry.build(panel, name, include_size=get(cfg, "features.include_size", False))
    report = features.registry.feature_report(df, meta)

    shared.io.write_parquet(df, out / "features.parquet")
    shared.io.write_json(meta, out / "feature_meta.json")
    shared.io.write_json(report, out / "feature_report.json")
    nonfinite = {k: v for k, v in report["nonfinite_counts"].items() if v}
    log.info(
        "feature set %s: %d rows x %d features; rows with all clustering features finite: %d; non-finite: %s",
        name,
        report["n_rows"],
        len(meta["features"]),
        report["rows_with_all_clustering_features_finite"],
        nonfinite or "none",
    )
    for flag, info in meta["data_quality"]["flags"].items():
        log.info("data-quality flag %s: %d rows, %d tickers", flag, info["rows"], info["tickers"])
    for flag, cols in meta["data_quality"]["skipped"].items():
        log.warning("data-quality flag %s skipped: panel lacks %s", flag, cols)
    for check in report["redundancy_checks"]:
        log.info(
            "redundancy %s vs %s: pearson %.3f spearman %.3f",
            check["a"],
            check["b"],
            check["pearson"],
            check["spearman"],
        )
    shared.runs.record_stage(run_dir, STAGE, cfg, started, feature_set=name, rows=report["n_rows"])
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
