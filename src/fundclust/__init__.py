"""Import aliases for the numbered stage packages under ``src/``.

Stage directories start with digits (``01_load_data``), which the ``import`` statement cannot name, so
this module maps short names to the stage ``code`` packages through :mod:`importlib`. Each
implementation exists once, in its stage folder; nothing is copied or symlinked (Windows creates
symlinks only with administrator rights or Developer Mode).

    from fundclust import features
    features.baseline12.build_baseline12(panel)

Aliases are resolved lazily on first attribute access, so importing ``fundclust`` does not import
every stage (and their heavy dependencies), and a stage that does not exist yet fails only when used.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_ALIASES = {
    "shared": "00_shared.code",
    "load": "01_load_data.code",
    "features": "02_features.code",
    "denoise": "03_denoise.code",
    "reduce": "04_reduce.code",
    "cluster": "05_cluster.code",
    "evaluate": "06_evaluate.code",
    "report": "07_report.code",
}

__all__ = sorted(_ALIASES)


def __getattr__(name: str) -> ModuleType:
    target = _ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'fundclust' has no attribute {name!r}; known aliases: {__all__}")
    try:
        module = importlib.import_module(target)
    except ModuleNotFoundError as exc:
        # Only translate the stage package itself being absent; a missing third-party dependency
        # inside an existing stage must surface as the original error.
        if exc.name is not None and target.startswith(exc.name):
            raise AttributeError(
                f"fundclust.{name} maps to src/{target.replace('.', '/')}, which does not exist (yet): {exc}"
            ) from exc
        raise
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ALIASES))
