"""One matplotlib style for every stage, and a ``savefig`` helper (PNG, dpi 150).

Colours come from the validated reference palette of the dataviz guidance used for this project
(light surface only, because figures are static PNGs embedded in light-themed Markdown):

- categorical slots in a fixed order; only the first three are validated for all-pairs comparison
  (scatter plots), the full eight for adjacent comparisons (bars, lines). Clusters beyond the palette
  are folded into one grey "other" colour by :func:`cluster_colors` instead of generating new hues;
- a single-hue blue ramp for magnitudes and a blue <-> red diverging pair with a grey midpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")  # headless: stages run from PowerShell and pytest without a display.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DPI = 150

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NOISE_GREY = "#c3c2b7"
OTHER_GREY = "#8a8984"

CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGING = ("#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a32a29")

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("fc_sequential", SEQUENTIAL_BLUE)
DIVERGING_CMAP = LinearSegmentedColormap.from_list("fc_diverging", DIVERGING)


def apply_style() -> None:
    """Set rcParams once per process; recessive axes and grid so data marks carry the contrast."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=list(CATEGORICAL)),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "image.cmap": "fc_sequential",
        }
    )
    for cmap in (SEQUENTIAL_CMAP, DIVERGING_CMAP):
        if cmap.name not in matplotlib.colormaps:
            matplotlib.colormaps.register(cmap)


def cluster_colors(labels: Iterable[int], max_named: int = len(CATEGORICAL)) -> Mapping[int, str]:
    """Map cluster ids to colours: largest clusters get palette slots in order, the rest share grey.

    Noise (``-1``) always gets the light ``NOISE_GREY``. Ordering by size keeps a cluster's colour
    stable when small clusters appear or vanish between runs with the same large clusters.
    """
    import pandas as pd

    counts = pd.Series(list(labels)).value_counts()
    ordered = [int(k) for k in counts.index if int(k) != -1]
    mapping: dict[int, str] = {-1: NOISE_GREY}
    for rank, label in enumerate(ordered):
        mapping[label] = CATEGORICAL[rank] if rank < max_named else OTHER_GREY
    return mapping


def savefig(fig: plt.Figure, path: str | Path, close: bool = True) -> Path:
    """Save as PNG at ``DPI`` with a tight box; parent folders are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
