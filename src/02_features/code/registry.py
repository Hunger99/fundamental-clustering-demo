"""Feature-set registry: build a named set, its metadata, and a numeric report.

``feature_meta.json`` lists ``features`` (every computed feature column), ``clustering_features`` (the
recommended clustering input) and ``auxiliary`` columns (flags that must not be clustered on), plus a
formula and reason per feature. Later stages read the column lists from it instead of hard-coding them.

Both sets also carry the data-quality flags of :mod:`.quality` as auxiliary columns. They are added here,
after the set's own builder, so the 12 baseline columns keep the sha256 that
``test_feature_set_matches_stored_checksum`` stores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import baseline12, fundamental, quality

FEATURE_SETS = ("baseline12", "fundamental_v2")
# Pairs reported explicitly because they are suspected redundancies (see README).
REDUNDANCY_CHECKS = {
    "baseline12": [("ros", "netmargin"), ("roe", "debt_eq"), ("pe", "price")],
    "fundamental_v2": [("roa", "ocf_to_assets"), ("quick", "current_ratio"), ("equity_ratio", "debt_to_assets")],
}


def _check(name: str) -> None:
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; expected one of {FEATURE_SETS}")


def required_columns(name: str, include_size: bool = False) -> list[str]:
    """Raw panel columns the feature set reads (used by the ``feature_columns`` missing policy)."""
    _check(name)
    if name == "baseline12":
        return list(baseline12.REQUIRED_RAW)
    return list(fundamental.required_raw(include_size))


def build(panel: pd.DataFrame, name: str, include_size: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return ``(features, meta)`` for the named set; ``include_size`` applies to ``fundamental_v2`` only.

    The data-quality flag columns are appended to either set and listed under ``meta["auxiliary"]``;
    ``meta["data_quality"]`` holds each rule, the rows and tickers it catches, and any skipped rule.
    """
    df, meta = _build_set(panel, name, include_size)
    flags, report = quality.data_quality_flags(panel.reset_index(drop=True))
    for col in flags.columns:
        df[col] = flags[col].to_numpy()
        meta["auxiliary"][col] = f"1 = {quality.FLAGS[col]['suspect']} suspect: {quality.FLAGS[col]['rule']}"
    meta["data_quality"] = report
    return df, meta


def _build_set(panel: pd.DataFrame, name: str, include_size: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    _check(name)
    if name == "baseline12":
        df = baseline12.build_baseline12(panel)
        meta = {
            "feature_set": name,
            "description": "The 12 ratios of the UMAP + DBSCAN baseline, in its fixed operation order.",
            "features": list(baseline12.COLUMNS),
            "clustering_features": list(baseline12.COLUMNS),
            "auxiliary": {},
            "formulas": dict(baseline12.FORMULAS),
            "reasons": {},
            "required_raw_columns": list(baseline12.REQUIRED_RAW),
        }
        return df, meta
    df = fundamental.build_fundamental_v2(panel, include_size=include_size)
    feats = fundamental.selected(include_size)
    meta = {
        "feature_set": name,
        "description": "Scale-free ratio features; valuation as yields; leverage defined for equity <= 0.",
        "include_size": bool(include_size),
        "features": [f.name for f in feats],
        "clustering_features": [f.name for f in feats if f.clustering],
        "auxiliary": dict(fundamental.AUXILIARY),
        "formulas": {f.name: f.formula for f in feats},
        "reasons": {f.name: f.reason for f in feats},
        "required_raw_columns": list(fundamental.required_raw(include_size)),
    }
    return df, meta


def feature_report(df: pd.DataFrame, meta: dict[str, Any], high_corr: float = 0.8) -> dict[str, Any]:
    """Non-finite counts, quantiles, strongly rank-correlated pairs and the named redundancy checks.

    Spearman is reported next to Pearson because the raw features have extreme tails (e.g. pb from
    -13,478 to 6,930): a handful of outliers can make Pearson near +-1 or near 0 on its own.
    """
    cols = meta["features"]
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    nonfinite = {c: int(v) for c, v in X.isna().sum().items()}
    quant = X.quantile([0.0, 0.01, 0.5, 0.99, 1.0]).T
    quant.columns = ["min", "p01", "median", "p99", "max"]
    spearman = X.corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rho = spearman.loc[a, b]
            if np.isfinite(rho) and abs(rho) >= high_corr:
                pairs.append({"a": a, "b": b, "spearman": float(rho)})
    checks = []
    for a, b in REDUNDANCY_CHECKS.get(meta["feature_set"], []):
        if a in X and b in X:
            both = X[[a, b]].dropna()
            ratio = (both[a] / both[b]).replace([np.inf, -np.inf], np.nan)
            checks.append(
                {
                    "a": a,
                    "b": b,
                    "n": int(len(both)),
                    "pearson": float(both[a].corr(both[b])),
                    "spearman": float(both[a].corr(both[b], method="spearman")),
                    "median_ratio_a_over_b": float(ratio.median()),
                }
            )
    rows_complete = int(X[meta["clustering_features"]].notna().all(axis=1).sum())
    return {
        "feature_set": meta["feature_set"],
        "n_rows": int(len(df)),
        "rows_with_all_clustering_features_finite": rows_complete,
        "nonfinite_counts": nonfinite,
        "quantiles": {c: {k: float(v) for k, v in quant.loc[c].items()} for c in cols},
        "high_spearman_pairs": pairs,
        "redundancy_checks": checks,
        "auxiliary_counts": {c: int(df[c].sum()) for c in meta.get("auxiliary", {}) if c in df},
    }
