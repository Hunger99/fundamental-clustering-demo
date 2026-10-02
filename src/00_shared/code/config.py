"""YAML configuration loading: ``configs/default.yaml`` with an optional variant deep-merged on top.

A variant file holds only the keys it changes. Mappings merge recursively; lists and scalars in the
variant replace the default wholesale, because merging lists element-wise (e.g. a stage list) would
make a variant unable to remove an entry.

The returned dict carries a ``_meta`` entry (config name and source files) so the provenance snapshot
records which files produced it; stage code should read settings with :func:`get`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import configs_dir

DEFAULT_NAME = "default"
_MISSING = object()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` merged over ``base`` (inputs are not mutated)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_variant(variant: str | Path | None) -> Path | None:
    """Map a variant name (``configs/<name>.yaml``) or an explicit path to a file; ``None`` = default only."""
    if variant is None or str(variant) in ("", DEFAULT_NAME):
        return None
    candidate = Path(variant)
    if candidate.suffix in (".yaml", ".yml") and candidate.is_file():
        return candidate.resolve()
    named = configs_dir() / f"{variant}.yaml"
    if named.is_file():
        return named
    raise FileNotFoundError(f"Config variant {variant!r} not found as a path or as {named}")


def load_config(variant: str | Path | None = None, default_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config and deep-merge ``variant`` over it."""
    default_file = Path(default_path) if default_path else configs_dir() / f"{DEFAULT_NAME}.yaml"
    cfg = load_yaml(default_file)
    sources = [str(default_file.resolve())]
    name = DEFAULT_NAME
    variant_file = resolve_variant(variant)
    if variant_file is not None:
        cfg = deep_merge(cfg, load_yaml(variant_file))
        sources.append(str(variant_file))
        name = variant_file.stem
    cfg["_meta"] = {"name": name, "sources": sources}
    return cfg


def get(cfg: dict[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    """Read ``a.b.c`` from nested dicts; without a default a missing key raises ``KeyError`` naming the path."""
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif default is _MISSING:
            raise KeyError(f"config key '{dotted}' is missing (failed at '{part}')")
        else:
            return default
    return node


def config_name(cfg: dict[str, Any]) -> str:
    return cfg.get("_meta", {}).get("name", DEFAULT_NAME)


def config_hash(cfg: dict[str, Any]) -> str:
    """sha256 of the settings, excluding ``_meta`` so the same settings from different files compare equal."""
    payload = {k: v for k, v in cfg.items() if k != "_meta"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
