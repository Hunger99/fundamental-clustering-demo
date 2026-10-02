"""Parquet and JSON helpers shared by all stages.

JSON writing converts numpy scalars/arrays, pandas timestamps and non-finite floats so that metric
dicts can be dumped directly; NaN/inf become ``null`` because strict JSON has no representation for
them and downstream readers (browsers, ``json.loads`` in other languages) reject the bare tokens.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write without the pandas index; artifacts carry an explicit ``row_id`` column instead."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact {path}; run the stage that produces it first.")
    return pd.read_parquet(path, columns=columns)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to plain JSON types; non-finite floats -> ``None``."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
