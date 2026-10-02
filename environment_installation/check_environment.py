"""Check the installed environment: Python version, imports and versions, data root, input checksums.

    environment_installation\\.venv\\Scripts\\python.exe environment_installation\\check_environment.py

Exits 1 when any check fails and prints one PASS/FAIL line per check, so it can gate a pipeline run.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# sha256 of fund.csv as read on 2026-10-01; a mismatch means the data changed and the numbers quoted in the
# READMEs (25,130 rows, cluster scores) need re-measuring.
EXPECTED_SHA256 = {
    "fund.csv": "4782078f5ae496bd5847a1472463a58f4420b4466ac2608e099338a89acd7cd9",
}
# import name -> distribution name
MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "xgboost": "xgboost",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "pytest": "pytest",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    results: list[tuple[bool, str]] = []
    results.append((sys.version_info[:2] == (3, 11), f"python {sys.version.split()[0]} (expected 3.11)"))
    for module, dist in MODULES.items():
        try:
            importlib.import_module(module)
            results.append((True, f"import {module} ({dist} {metadata.version(dist)})"))
        except Exception as exc:  # report every broken import, not only the first
            results.append((False, f"import {module}: {type(exc).__name__}: {exc}"))
    try:
        from fundclust import shared

        results.append((True, "import fundclust.shared (src/ on sys.path)"))
        root = shared.paths.data_root()
        results.append((True, f"data root {root}"))
        for name, expected in EXPECTED_SHA256.items():
            path = shared.paths.raw_file(name)
            actual = _sha256(path)
            results.append((actual == expected, f"sha256 {name} {actual[:16]}... (expected {expected[:16]}...)"))
        indicators = shared.paths.raw_file("SHARADAR_INDICATORS_a3407c5c2ec46991d2b7ac667785d0d1.csv")
        results.append((True, f"indicator definitions {indicators.name}"))
    except Exception as exc:
        results.append((False, f"{type(exc).__name__}: {exc}"))

    for ok, text in results:
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    failed = sum(not ok for ok, _ in results)
    print(f"{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
