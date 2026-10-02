"""Run directories and their provenance records.

Every run lives in ``<root>/<config>--<dataset>--<UTC %Y%m%dT%H%M%SZ>/`` with ``artifacts/``, ``logs/``
and ``provenance/``. Production and experiment runs use ``outputs/archive/runs/``; the baseline passes
its own root. Provenance files:

    provenance/config.yaml        resolved configuration snapshot (written once, at creation)
    provenance/run.json           argv, source folder label, git HEAD + dirty flag, python + package versions
    provenance/inputs.json        sha256 of every input file a stage read, keyed by stage
    provenance/stages.json        one entry per completed stage (config hash, UTC start/end, argv)

A stage run against an existing run dir with a different resolved config is refused
(:class:`ConfigMismatchError`): mixing settings inside one run would make its provenance false.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import config_hash, config_name, get
from .paths import PROJECT_ROOT, runs_root

STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
RUN_SUBDIRS = ("artifacts", "logs", "provenance")
KEY_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "numba",
    "pynndescent",
    "xgboost",
    "matplotlib",
    "seaborn",
    "plotly",
    "pyarrow",
    "pyyaml",
)


class ConfigMismatchError(RuntimeError):
    """Raised when a stage would write into a run created with different settings."""


def utc_stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(STAMP_FORMAT)


def run_name(config: str, dataset: str, now: datetime | None = None) -> str:
    return f"{config}--{dataset}--{utc_stamp(now)}"


def make_run_dir(config: str, dataset: str, root: Path | None = None, now: datetime | None = None) -> Path:
    """Create a fresh run directory and its standard subfolders.

    Two runs started in the same UTC second would collide; the second gets a ``_2`` (``_3``...) suffix
    instead of overwriting, because an existing run dir may hold verified results.
    """
    base = Path(root) if root is not None else runs_root()
    base.mkdir(parents=True, exist_ok=True)
    name = run_name(config, dataset, now)
    path = base / name
    suffix = 2
    while path.exists():
        path = base / f"{name}_{suffix}"
        suffix += 1
    path.mkdir(parents=False)
    for sub in RUN_SUBDIRS:
        (path / sub).mkdir()
    return path


def stage_dir(run_dir: Path, stage: str) -> Path:
    """``<run>/artifacts/<stage>/``, created on demand."""
    out = Path(run_dir) / "artifacts" / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions(names: Iterable[str] = KEY_PACKAGES) -> dict[str, str | None]:
    """Installed versions; ``None`` marks a package that is not installed rather than failing the run."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_state(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """HEAD commit and dirty flag; git being unavailable is recorded, not raised, so runs still work."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()

    try:
        head = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        return {"available": True, "head": head, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "head": None, "dirty": None, "error": str(exc)}


def source_label(path: str | Path) -> str:
    """Name the code folder a run came from: ``src``, ``experiment/<folder>``, ``baselines/<method>``."""
    try:
        rel = Path(path).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return str(Path(path).resolve())
    parts = rel.parts
    if parts and parts[0] in ("experiment", "baselines") and len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else "."


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_provenance(
    run_dir: Path,
    cfg: Mapping[str, Any],
    *,
    source: str,
    argv: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write ``config.yaml`` and ``run.json`` for a new run (called once, right after creation)."""
    prov = Path(run_dir) / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    with open(prov / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, allow_unicode=True)
    record = {
        "run_dir": str(Path(run_dir).resolve()),
        "created_utc": utc_stamp(),
        "config_name": config_name(dict(cfg)),
        "config_sha256": config_hash(dict(cfg)),
        "source": source,
        "argv": list(argv if argv is not None else sys.argv),
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if extra:
        record["extra"] = dict(extra)
    _write_json(prov / "run.json", record)


def record_inputs(run_dir: Path, stage: str, files: Mapping[str, str | Path]) -> dict[str, dict[str, str]]:
    """Append sha256 + path of each input file a stage read to ``provenance/inputs.json``."""
    path = Path(run_dir) / "provenance" / "inputs.json"
    data = _read_json(path, {})
    entry = {label: {"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for label, p in files.items()}
    data[stage] = entry
    _write_json(path, data)
    return entry


def record_stage(run_dir: Path, stage: str, cfg: Mapping[str, Any], started_utc: str, **info: Any) -> None:
    """Append a completed-stage entry to ``provenance/stages.json``."""
    path = Path(run_dir) / "provenance" / "stages.json"
    data = _read_json(path, [])
    data.append(
        {
            "stage": stage,
            "config_sha256": config_hash(dict(cfg)),
            "started_utc": started_utc,
            "finished_utc": utc_stamp(),
            "argv": list(sys.argv),
            **info,
        }
    )
    _write_json(path, data)


def open_run(
    run_dir: str | Path | None,
    cfg: dict[str, Any],
    *,
    source: str,
    root: Path | None = None,
    argv: list[str] | None = None,
) -> Path:
    """Return a usable run dir: create one (with provenance) when ``run_dir`` is None, else validate it.

    An existing run dir must already hold ``provenance/config.yaml`` whose settings hash equals ``cfg``'s;
    otherwise :class:`ConfigMismatchError` is raised. A directory without provenance (e.g. made by hand)
    gets provenance written now, so later stages can still be checked against it.
    """
    if run_dir is None:
        path = make_run_dir(config_name(cfg), get(cfg, "data.dataset_id"), root=root)
        write_provenance(path, cfg, source=source, argv=argv)
        return path
    path = Path(run_dir).resolve()
    if not path.is_dir():
        # A typo in --run-dir must not silently start a new, half-empty run somewhere else.
        raise FileNotFoundError(f"run dir {path} does not exist; omit --run-dir to create a new run")
    snapshot = path / "provenance" / "config.yaml"
    if not snapshot.is_file():
        for sub in RUN_SUBDIRS:
            (path / sub).mkdir(parents=True, exist_ok=True)
        write_provenance(path, cfg, source=source, argv=argv)
        return path
    with open(snapshot, encoding="utf-8") as fh:
        stored = yaml.safe_load(fh) or {}
    if config_hash(stored) != config_hash(cfg):
        raise ConfigMismatchError(
            f"{path} was created with different settings ({snapshot}); start a new run dir or pass the "
            "same --config that created it."
        )
    return path
