"""Stage 01 command line: write ``artifacts/01_load_data/`` into a run directory.

    python -m 01_load_data.utils.cli [--run-dir DIR] [--config NAME|PATH]

Without ``--run-dir`` a new run is created under ``outputs/archive/runs/``. Outputs:
``panel.parquet``, ``missing_report.json``, ``indicator_definitions.parquet``.
"""

from __future__ import annotations

import argparse
import sys

from fundclust import load, shared

STAGE = "01_load_data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"{STAGE}.utils.cli", description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=None, help="existing run dir; omitted = create a new one")
    parser.add_argument("--config", default=None, help="variant name under configs/ or a YAML path")
    args = parser.parse_args(argv)

    cfg = shared.config.load_config(args.config)
    get = shared.config.get
    started = shared.runs.utc_stamp()
    run_dir = shared.runs.open_run(args.run_dir, cfg, source="src", argv=list(sys.argv))
    log = shared.logs.setup_logging(run_dir, STAGE)
    out = shared.runs.stage_dir(run_dir, STAGE)

    fund_path = shared.paths.raw_file(get(cfg, "data.fund_file"))
    defs_path = shared.paths.raw_file(get(cfg, "data.indicators_file"))
    inputs = shared.runs.record_inputs(run_dir, STAGE, {"fund_csv": fund_path, "indicators_csv": defs_path})
    log.info("inputs: %s", {k: v["sha256"][:12] for k, v in inputs.items()})

    policy = get(cfg, "missing.policy")
    required = None
    if policy == "feature_columns":
        # Imported lazily: only this policy needs to know which raw columns the feature set reads.
        from fundclust import features

        required = features.registry.required_columns(
            get(cfg, "features.set"), include_size=get(cfg, "features.include_size", False)
        )
    panel, report = load.load.load_panel(
        fund_path,
        drop_dates=get(cfg, "data.drop_calendardates"),
        policy=policy,
        required_columns=required,
        probe_tickers=get(cfg, "missing.financial_probe_tickers", []),
    )
    defs = load.load.read_indicator_definitions(defs_path, panel.columns)

    shared.io.write_parquet(panel, out / "panel.parquet")
    shared.io.write_parquet(defs, out / "indicator_definitions.parquet")
    shared.io.write_json(report, out / "missing_report.json")
    hyp = report["financial_firm_hypothesis"]["probe_summary"]
    log.info(
        "policy=%s kept %d/%d rows, %d tickers; financial probes present=%d dropped=%d kept=%s",
        policy,
        report["rows"]["kept"],
        report["rows"]["after_date_filter"],
        report["tickers"]["kept"],
        hyp["present_in_file"],
        hyp["dropped_entirely"],
        hyp["kept_tickers"],
    )
    shared.runs.record_stage(run_dir, STAGE, cfg, started, rows=report["rows"]["kept"])
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
