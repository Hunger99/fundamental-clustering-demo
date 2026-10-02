"""Deterministic names for the clustering's archetypes and the principal axes.

Cluster ids are ordered by size, so they can move when a refit changes which group is larger. Names come
from the cluster medians of the unscaled ratios through a fixed rule, so the same business model gets the
same name in every run that finds it, and a reader never has to trust a hand-written id-to-name table.

Archetype rule (:func:`name_archetypes`), applied to one row of medians per cluster:

    1. a cluster is loss-making when its median ``roa`` is below 0, profitable otherwise;
    2. among loss-making clusters, the one with the lowest median ``asset_turnover`` is
       "pre-revenue development stage" when that turnover is below 0.05 per quarter (sales under 5% of
       assets); every other loss-making cluster is "loss-making cash burners";
    3. among profitable clusters, the one with the highest median ``asset_turnover`` is
       "thin-margin high-turnover"; of the rest, the one with the highest median ``cash_to_assets`` is
       "cash-rich profitable"; the others are "mature profitable leveraged".

A name that falls to more than one cluster gets a numeric suffix in cluster-id order (" 2", " 3"). The
rule only reads signs and orderings except for the 0.05 turnover bound. On the production run of
2026-10-01 the medians that decide are: roa -0.081 and -0.401 for the two loss-making clusters against
0.048-0.102; asset turnover 0.016 for the pre-revenue cluster against 0.144 for the cash burners; 0.341
for the high-turnover cluster against 0.129 and 0.198; cash 17.8% of assets against 4.0%.

Axis rule (:func:`name_axes`): each candidate name has a signature of features and signs; its score on a
component is the mean of ``sign * correlation loading`` over the signature. Names are matched one to one
to the first r components by the largest total absolute score (Hungarian assignment), so the order of the
components never decides a name. A component whose best score is below 0.4 in absolute value stays
unnamed. A negative score means the component points the other way; the report then flips its sign so
that higher always reads as more of the named quantity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PRE_REVENUE = "pre-revenue development stage"
CASH_BURNERS = "loss-making cash burners"
HIGH_TURNOVER = "thin-margin high-turnover"
CASH_RICH = "cash-rich profitable"
LEVERAGED = "mature profitable leveraged"
ARCHETYPE_NAMES = (LEVERAGED, HIGH_TURNOVER, CASH_RICH, CASH_BURNERS, PRE_REVENUE)
# Quarterly revenue / average assets below which a loss-making group counts as pre-revenue. Chosen between
# the measured medians of the two loss-making archetypes (0.016 pre-revenue, 0.144 cash burners).
PRE_REVENUE_TURNOVER = 0.05
REQUIRED = ("roa", "asset_turnover", "cash_to_assets")

AXIS_SIGNATURES: dict[str, dict[str, int]] = {
    "profitability": {"operating_margin": 1, "netmargin": 1, "roa": 1},
    "balance-sheet strength": {"equity_ratio": 1, "debt_to_assets": -1, "current_ratio": 1},
    "value": {"book_to_market": 1, "sales_to_price": 1},
    "asset turnover": {"asset_turnover": 1},
}
# Smallest mean signed correlation that earns a name. On the production run the matched scores are
# 0.62-0.78 and the largest unmatched one is 0.46 (PC1 on balance-sheet strength, through its liquidity
# loadings), so the one-to-one matching keeps names apart and 0.4 only stops a name on weak loadings.
AXIS_MIN_SCORE = 0.4


def cluster_medians(features: pd.DataFrame, labels: Any, columns: Sequence[str], noise_label: int = -1) -> pd.DataFrame:
    """Median of each feature per cluster over its rows (noise excluded); index = cluster id, sorted."""
    frame = features[list(columns)].copy()
    frame["_label"] = np.asarray(labels)
    frame = frame[frame["_label"] != noise_label]
    return frame.groupby("_label", sort=True)[list(columns)].median().rename_axis("label")


def name_archetypes(medians: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """``{cluster id: {"name", "step", "role"}}`` from the rule in the module docstring.

    ``medians`` has one row per cluster (index = id) and at least the columns ``roa``, ``asset_turnover``
    and ``cash_to_assets`` in ratio units. Ties in an ordering go to the smaller cluster id.
    """
    missing = [c for c in REQUIRED if c not in medians.columns]
    if missing:
        raise KeyError(f"cluster medians lack {missing}; the naming rule needs {list(REQUIRED)}")
    med = medians.sort_index()
    ids = [int(i) for i in med.index]
    roa = med["roa"].to_numpy(dtype=float)
    turnover = med["asset_turnover"].to_numpy(dtype=float)
    cash = med["cash_to_assets"].to_numpy(dtype=float)
    out: dict[int, dict[str, Any]] = {}

    loss = [i for i, v in enumerate(roa) if v < 0]
    prof = [i for i, v in enumerate(roa) if not v < 0]
    # Stable sorts on the id order already in place make ties go to the smaller id.
    loss_by_turnover = sorted(loss, key=lambda i: turnover[i])
    for rank, i in enumerate(loss_by_turnover):
        if rank == 0 and turnover[i] < PRE_REVENUE_TURNOVER:
            out[ids[i]] = {"name": PRE_REVENUE, "role": "loss-making",
                           "step": f"roa {roa[i]:.3f} < 0; lowest asset turnover {turnover[i]:.3f} < {PRE_REVENUE_TURNOVER}"}
        else:
            out[ids[i]] = {"name": CASH_BURNERS, "role": "loss-making",
                           "step": f"roa {roa[i]:.3f} < 0; asset turnover {turnover[i]:.3f}"}
    remaining = list(prof)
    if remaining:
        i = sorted(remaining, key=lambda j: -turnover[j])[0]
        out[ids[i]] = {"name": HIGH_TURNOVER, "role": "profitable",
                       "step": f"roa {roa[i]:.3f} >= 0; highest asset turnover {turnover[i]:.3f}"}
        remaining.remove(i)
    if remaining:
        i = sorted(remaining, key=lambda j: -cash[j])[0]
        out[ids[i]] = {"name": CASH_RICH, "role": "profitable",
                       "step": f"roa {roa[i]:.3f} >= 0; highest cash/assets {cash[i]:.3f} of the rest"}
        remaining.remove(i)
    for i in remaining:
        out[ids[i]] = {"name": LEVERAGED, "role": "profitable",
                       "step": f"roa {roa[i]:.3f} >= 0; neither highest turnover nor highest cash"}

    seen: dict[str, int] = {}
    for cid in sorted(out):
        base = out[cid]["name"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            out[cid]["name"] = f"{base} {seen[base]}"
    return dict(sorted(out.items()))


def naming_checks(medians: pd.DataFrame, names: Mapping[int, Mapping[str, Any]]) -> dict[str, bool | None]:
    """Properties the names promise beyond the variables the rule reads; ``None`` when a name is absent.

    The rule picks names from roa, turnover and cash only. These checks say whether the other words in
    each name hold on the same medians: the high-turnover group has the thinnest net margin among
    profitable groups, the leveraged group the highest debt to assets among them, and the pre-revenue
    group the highest quick ratio overall.
    """
    by_name = {v["name"]: k for k, v in names.items()}
    prof = [k for k, v in names.items() if v["role"] == "profitable"]
    out: dict[str, bool | None] = {}

    def _extreme(name: str, col: str, pool: list[int], highest: bool) -> bool | None:
        if name not in by_name or col not in medians.columns or not pool:
            return None
        vals = medians.loc[pool, col]
        target = vals.idxmax() if highest else vals.idxmin()
        return bool(int(target) == by_name[name])

    out["high_turnover_has_lowest_profitable_netmargin"] = _extreme(HIGH_TURNOVER, "netmargin", prof, False)
    out["leveraged_has_highest_profitable_debt_to_assets"] = _extreme(LEVERAGED, "debt_to_assets", prof, True)
    out["pre_revenue_has_highest_quick_ratio"] = _extreme(PRE_REVENUE, "quick", list(names), True)
    return out


def name_axes(
    corr_loadings: pd.DataFrame,
    r: int,
    signatures: Mapping[str, Mapping[str, int]] = AXIS_SIGNATURES,
    min_score: float = AXIS_MIN_SCORE,
) -> list[dict[str, Any]]:
    """One entry per component ``PC1..PCr``: ``pc``, ``name``, ``sign`` (+1/-1), ``score`` and all ``scores``.

    ``corr_loadings`` is features x components (correlation of each feature with each component's scores).
    A signature feature absent from the table is skipped; a signature with none present scores 0.
    """
    pcs = [f"PC{k + 1}" for k in range(int(r))]
    names = list(signatures)
    S = np.zeros((len(pcs), len(names)))
    for j, name in enumerate(names):
        terms = [(f, s) for f, s in signatures[name].items() if f in corr_loadings.index]
        for k, pc in enumerate(pcs):
            if terms:
                S[k, j] = float(np.mean([s * corr_loadings.loc[f, pc] for f, s in terms]))
    rows, cols = linear_sum_assignment(-np.abs(S))
    match = dict(zip(rows, cols))
    out = []
    for k, pc in enumerate(pcs):
        entry: dict[str, Any] = {"pc": pc, "scores": {n: float(S[k, j]) for j, n in enumerate(names)}}
        j = match.get(k)
        if j is not None and abs(S[k, j]) >= min_score:
            entry.update(name=names[j], sign=1 if S[k, j] >= 0 else -1, score=float(S[k, j]))
        else:
            entry.update(name=None, sign=1, score=float(S[k, j]) if j is not None else float("nan"))
        out.append(entry)
    return out


def display_name(name: str) -> str:
    """Sentence case for titles and tables: first letter upper, the rest as written."""
    return name[:1].upper() + name[1:] if name else name
