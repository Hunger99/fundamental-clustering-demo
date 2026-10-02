"""Shared command line used by the launchers: create a run dir, list configured stages, check them.

    python -m 00_shared.utils.cli new-run [--config NAME|PATH]      prints the new run dir path
    python -m 00_shared.utils.cli stages  [--config NAME|PATH]      prints pipeline.stages, one per line
    python -m 00_shared.utils.cli check-stages [--config NAME|PATH] exits 2 if a listed stage has no CLI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..code import config as cfgmod
from ..code import paths, runs


def stage_cli_path(stage: str) -> Path:
    return paths.PROJECT_ROOT / "src" / stage / "utils" / "cli.py"


def run_source(cfg: dict) -> str:
    """Provenance source label: the study or baseline folder that holds the variant config, else ``src``.

    Studies launch their variants through these same launchers, so labelling every new run ``src`` would
    record a study run as a production one.
    """
    sources = cfg.get("_meta", {}).get("sources", [])
    if len(sources) > 1:
        label = runs.source_label(sources[-1])
        if label.split("/")[0] in ("experiment", "baselines"):
            return label
    return "src"


def missing_stages(stages: list[str]) -> list[str]:
    """Stages listed in config whose ``src/<stage>/utils/cli.py`` does not exist."""
    return [s for s in stages if not stage_cli_path(s).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="00_shared.utils.cli", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["new-run", "stages", "check-stages"])
    parser.add_argument("--config", default=None, help="variant name under configs/ or a YAML path")
    args = parser.parse_args(argv)
    cfg = cfgmod.load_config(args.config)
    stages = list(cfgmod.get(cfg, "pipeline.stages"))

    if args.command == "new-run":
        run_dir = runs.open_run(None, cfg, source=run_source(cfg), argv=list(sys.argv))
        print(run_dir)
        return 0
    if args.command == "stages":
        print("\n".join(stages))
        return 0
    missing = missing_stages(stages)
    if missing:
        print(
            "pipeline.stages lists stages without an implementation: "
            + ", ".join(f"{s} (expected {stage_cli_path(s)})" for s in missing),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
