"""Project-root and external-data-root resolution.

The data root is machine-specific and lives outside the repository, so it is never written into
``src/`` or ``configs/``. Resolution order:

1. the ``FC_DATA_DIR`` environment variable;
2. ``FC_DATA_DIR=...`` in ``environment_installation/local.env`` (git-ignored; see ``local.env.example``).

A data root that resolves inside the repository is rejected: the layout requires data to be external,
and a git-ignored folder inside the repo would silently violate that.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "FC_DATA_DIR"

# src/00_shared/code/paths.py -> parents[3] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ENV = PROJECT_ROOT / "environment_installation" / "local.env"

FUND_CSV = "fund.csv"
INDICATORS_CSV = "SHARADAR_INDICATORS_a3407c5c2ec46991d2b7ac667785d0d1.csv"


class DataRootError(RuntimeError):
    """Raised when the external data root cannot be resolved or is unusable."""


def project_root() -> Path:
    return PROJECT_ROOT


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ``#`` comments and blank lines are ignored, quotes are stripped."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def data_root(must_exist: bool = True) -> Path:
    """Return the external data root.

    Raises :class:`DataRootError` with the two places to fix when neither source defines it, when the
    directory is missing (``must_exist``), or when it points inside the repository.
    """
    value = os.environ.get(ENV_VAR, "").strip()
    source = f"environment variable {ENV_VAR}"
    if not value:
        value = read_env_file(LOCAL_ENV).get(ENV_VAR, "").strip()
        source = str(LOCAL_ENV)
    if not value:
        raise DataRootError(
            f"{ENV_VAR} is not set. Set the environment variable or add '{ENV_VAR}=<path>' to "
            f"{LOCAL_ENV} (template: {LOCAL_ENV.with_name('local.env.example')})."
        )
    root = Path(value).expanduser().resolve()
    if must_exist and not root.is_dir():
        raise DataRootError(f"{ENV_VAR}={root} (from {source}) does not exist or is not a directory.")
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise DataRootError(f"{ENV_VAR}={root} is inside the repository; data must stay external.")
    return root


def raw_dir() -> Path:
    return data_root() / "raw"


def raw_file(name: str) -> Path:
    """Return ``<data root>/raw/<name>``, raising :class:`DataRootError` if it is missing."""
    path = raw_dir() / name
    if not path.is_file():
        raise DataRootError(f"Expected input file is missing: {path}")
    return path


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


def runs_root() -> Path:
    """Production and experiment runs: ``outputs/archive/runs/``."""
    return PROJECT_ROOT / "outputs" / "archive" / "runs"


def bank_dir(group: str = "clustering") -> Path:
    return PROJECT_ROOT / "outputs" / "bank" / group
