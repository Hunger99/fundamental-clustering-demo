"""Logging setup: one console handler plus a per-run file under ``<run>/logs/``."""

from __future__ import annotations

import logging
import time
from pathlib import Path

# Timestamps are UTC with a Z suffix so log lines line up with the UTC run-directory names.
FORMAT = "%(asctime)sZ %(levelname)s %(name)s: %(message)s"
ROOT_LOGGER = "fundclust"


def setup_logging(run_dir: str | Path | None, name: str, level: int = logging.INFO) -> logging.Logger:
    """Return logger ``fundclust.<name>`` writing to stderr and, with a run dir, ``logs/<name>.log``.

    Handlers are attached to the shared ``fundclust`` parent and replaced on each call, so running
    several stages in one interpreter (tests, the pipeline) neither duplicates lines nor keeps writing
    into the previous run's log file.
    """
    parent = logging.getLogger(ROOT_LOGGER)
    parent.setLevel(level)
    parent.propagate = False
    for handler in list(parent.handlers):
        parent.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    parent.addHandler(console)
    if run_dir is not None:
        log_dir = Path(run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        parent.addHandler(file_handler)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
