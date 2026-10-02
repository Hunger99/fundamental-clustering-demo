"""Numbers behind the report figures, computed from one run's artifacts.

Every function is pure (arrays and frames in, frames and dicts out) so the tests can check it on
synthetic panels. Resampling is always over whole companies: a company's quarters are near copies (rank
ICC about 0.76 on this panel), so a row bootstrap would give intervals several times too narrow.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from fundclust import cluster, evaluate, reduce


def quarter_label(q: int) -> str:
    """``2020Q2`` for the quarter index ``year * 4 + quarter - 1``."""
    return f"{q // 4}Q{q % 4 + 1}"


def project_centers(centers: Any, pca_json: Mapping[str, Any], columns: Sequence[str], r: int) -> np.ndarray:
    """PC scores of points given in the matrix PCA was fitted on (e.g. k-means centres), first ``r`` PCs.

    Uses the stored ``mean``, ``scale`` and unit ``loadings`` of ``pca.json``; ``columns`` names the
    points' columns so a different column order cannot silently misalign them.
    """
    feats = list(pca_json["features"])
    C = pd.DataFrame(np.atleast_2d(np.asarray(centers, dtype=np.float64)), columns=list(columns))[feats].to_numpy()
    mean = np.asarray(pca_json["mean"], dtype=np.float64)
    scale = np.asarray(pca_json["scale"], dtype=np.float64)
    load = pd.DataFrame(pca_json["loadings"]).T.loc[feats]
    U = load[[f"PC{k + 1}" for k in range(int(r))]].to_numpy(dtype=np.float64)
    return ((C - mean) / scale) @ U


def ticker_medians(frame: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    """One row per ticker: the median of each value column over its rows."""
    return frame.groupby("ticker", sort=True)[list(value_cols)].median()


def sector_mix(modal: pd.Series, sectors: pd.Series) -> dict[str, pd.DataFrame]:
    """Sector composition of each archetype on companies (each in its modal archetype).

    Returns ``counts`` (archetype x sector, companies with a known sector), ``share`` (row shares),
    ``lift`` (share in the archetype / share in the panel) and ``unlabelled`` (companies without a sector
    per archetype). Sector labels are current listings, so delisted companies are the unlabelled ones.
    """
    df = pd.DataFrame({"label": modal.to_numpy()}, index=modal.index)
    df["sector"] = sectors.reindex(df.index).to_numpy(dtype=object)
    known = df[df["sector"].notna()]
    counts = pd.crosstab(known["label"], known["sector"]).sort_index()
    share = counts.div(counts.sum(axis=1), axis=0)
    panel_share = counts.sum(axis=0) / counts.to_numpy().sum()
    lift = share.div(panel_share, axis=1)
    unlabelled = df[df["sector"].isna()].groupby("label").size().reindex(counts.index, fill_value=0)
    return {"counts": counts, "share": share, "lift": lift, "unlabelled": unlabelled}


def transitions(labels: Any, tickers: Any, dates: Any) -> pd.DataFrame:
    """Rows with the same ticker's label one calendar quarter earlier: ``ticker, q, label, prev``.

    A reporting gap breaks the pair: only consecutive quarters count as a move or a stay.
    """
    df = pd.DataFrame({"ticker": np.asarray(tickers), "q": evaluate.quarter_index(dates), "label": np.asarray(labels)})
    df = df.drop_duplicates(["ticker", "q"], keep=False)
    prev = df.rename(columns={"label": "prev"}).assign(q=lambda d: d["q"] + 1)
    return df.merge(prev, on=["ticker", "q"], how="inner")


def _nanq(values: np.ndarray, q: float) -> Any:
    """Column quantile ignoring NaN; an all-NaN column (no company ever at risk) stays NaN without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(values, q, axis=0)


def _ticker_bootstrap_weights(n_tickers: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_tickers, np.full(n_tickers, 1.0 / n_tickers), size=n_boot).astype(np.float64)


def migration_rates(
    labels: Any,
    tickers: Any,
    dates: Any,
    loss_labels: Sequence[int],
    base_quarters: tuple[int, int],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Share of companies in a profitable archetype at t - 1 that sit in a loss-making archetype at t.

    Per quarter t: ``moved / at_risk`` with a ticker-bootstrap CI (multinomial company weights, the same
    draw for every quarter so ratios between quarters get a joint interval). The base rate pools every
    quarter in ``base_quarters`` (inclusive quarter indices). ``ratio_to_base`` per quarter carries its own
    bootstrap CI. Noise rows (label -1) are neither at risk nor movers.
    """
    tr = transitions(labels, tickers, dates)
    loss = set(int(x) for x in loss_labels)
    tr = tr[(tr["label"] != -1) & (tr["prev"] != -1)]
    at_risk = ~tr["prev"].isin(loss)
    tr = tr.assign(at_risk=at_risk.astype(float), moved=(at_risk & tr["label"].isin(loss)).astype(float))
    A = tr.pivot_table(index="ticker", columns="q", values="at_risk", aggfunc="sum", fill_value=0.0)
    M = tr.pivot_table(index="ticker", columns="q", values="moved", aggfunc="sum", fill_value=0.0).reindex_like(A)
    quarters = [int(q) for q in A.columns]
    base_cols = [j for j, q in enumerate(quarters) if base_quarters[0] <= q <= base_quarters[1]]
    a, m = A.to_numpy(), M.fillna(0.0).to_numpy()
    W = _ticker_bootstrap_weights(len(A), n_boot, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = m.sum(0) / a.sum(0)
        boot = (W @ m) / (W @ a)
        base = m[:, base_cols].sum() / a[:, base_cols].sum()
        base_boot = (W @ m[:, base_cols]).sum(1) / (W @ a[:, base_cols]).sum(1)
        ratio_boot = boot / base_boot[:, None]
    # A draw without any company at risk in a quarter gives 0/0; it is left NaN and skipped by the quantiles.
    ratio_boot[~np.isfinite(ratio_boot)] = np.nan
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rate / base if base > 0 else np.full(len(quarters), np.nan)
    table = pd.DataFrame({
        "q": quarters,
        "quarter": [quarter_label(q) for q in quarters],
        "at_risk": a.sum(0).astype(int),
        "moved": m.sum(0).astype(int),
        "rate": rate,
        "ci_low": _nanq(boot, lo),
        "ci_high": _nanq(boot, hi),
        "ratio_to_base": ratio,
        "ratio_ci_low": _nanq(ratio_boot, lo),
        "ratio_ci_high": _nanq(ratio_boot, hi),
        "in_base": [base_quarters[0] <= q <= base_quarters[1] for q in quarters],
    })
    base_single = table.loc[table["in_base"], "rate"]
    return {
        "table": table,
        "base_rate": float(base),
        "base_ci": [float(_nanq(base_boot, lo)), float(_nanq(base_boot, hi))],
        "base_quarters": [quarter_label(base_quarters[0]), quarter_label(base_quarters[1])],
        "base_single_quarter_range": [float(base_single.min()), float(base_single.max())] if len(base_single) else None,
        "n_tickers": int(len(A)),
        "n_boot": int(n_boot),
    }


def migration_by_sector(
    labels: Any,
    tickers: Any,
    dates: Any,
    loss_labels: Sequence[int],
    sectors: pd.Series,
    quarter: int,
    base_quarters: tuple[int, int],
) -> pd.DataFrame:
    """Move rate into a loss-making archetype per sector, in one quarter and pooled over the base quarters.

    Every sector with at least one company at risk in either period is listed with its counts, because
    sector-level counts are small (Energy has about 25 companies at risk per quarter) and a rate without
    its denominator would overstate what is known. Companies without a sector label form "no sector".
    """
    tr = transitions(labels, tickers, dates)
    loss = set(int(x) for x in loss_labels)
    tr = tr[(tr["label"] != -1) & (tr["prev"] != -1) & ~tr["prev"].isin(loss)]
    tr = tr.assign(moved=tr["label"].isin(loss).astype(int),
                   sector=sectors.reindex(tr["ticker"].to_numpy()).fillna("no sector").to_numpy())
    shock = tr[tr["q"] == quarter].groupby("sector")["moved"].agg(at_risk="size", moved="sum")
    in_base = (tr["q"] >= base_quarters[0]) & (tr["q"] <= base_quarters[1])
    base = tr[in_base].groupby("sector")["moved"].agg(base_at_risk="size", base_moved="sum")
    out = shock.join(base, how="outer").fillna(0).astype(int)
    out["rate"] = out["moved"] / out["at_risk"].where(out["at_risk"] > 0)
    out["base_rate"] = out["base_moved"] / out["base_at_risk"].where(out["base_at_risk"] > 0)
    return out.sort_values(["at_risk", "base_at_risk"], ascending=False).reset_index()


def clip_shares(features: pd.DataFrame, labels: Any, lo: Mapping[str, float], hi: Mapping[str, float]) -> pd.DataFrame:
    """Share of each archetype's rows strictly beyond the stage-03 winsor bounds, per feature.

    Stage 03 clipped these rows to the bound, so the clusterer could not tell them apart, and the group's
    median in raw units is a rough level when the share is large. Values equal to a bound are not counted:
    a bound of 0 (debt to assets) is reached by every company without debt, which loses nothing.
    """
    cols = [c for c in lo if c in features.columns]
    f = features[cols]
    at = (f.lt(pd.Series(lo)[cols]) | f.gt(pd.Series(hi)[cols])).astype(float)
    at["_label"] = np.asarray(labels)
    return at[at["_label"] != -1].groupby("_label").mean().rename_axis("label")


def market_cap_by_archetype(labels: Any, tickers: Any, market_cap: Any) -> pd.Series:
    """Median over companies of each company's median market cap in the archetype's quarters."""
    df = pd.DataFrame({"label": np.asarray(labels), "ticker": np.asarray(tickers),
                       "cap": np.asarray(market_cap, dtype=np.float64)})
    df = df[(df["label"] != -1) & np.isfinite(df["cap"]) & (df["cap"] > 0)]
    return df.groupby(["label", "ticker"])["cap"].median().groupby("label").median()


def quarter_adjusted_moves(fwd: Any, dates: Any) -> np.ndarray:
    """``|r - median_t(r)|``: size of the next-quarter log price change beyond that quarter's market median.

    ``r`` is NaN where the next quarter is not exactly one calendar quarter later; those rows stay NaN.
    Subtracting the cross-sectional median removes the market-wide move of the quarter (2020Q1 alone fell
    by more than most companies' typical move), so archetypes are compared on their own risk.
    """
    r = pd.Series(np.asarray(fwd, dtype=np.float64))
    q = pd.Series(evaluate.quarter_index(dates))
    med = r.groupby(q).transform("median")
    return np.abs(r - med).to_numpy()


def risk_by_archetype(
    labels: Any, tickers: Any, moves: Any, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per archetype: rows, companies, median quarter-adjusted move and its ticker-bootstrap 95% CI."""
    df = pd.DataFrame({"label": np.asarray(labels), "ticker": np.asarray(tickers), "move": np.asarray(moves)})
    df = df[(df["label"] != -1) & np.isfinite(df["move"])]
    rows = []
    for lab, g in df.groupby("label", sort=True):
        lo, hi = evaluate.ticker_bootstrap_median_ci(g["move"].to_numpy(), g["ticker"].to_numpy(), n_boot=n_boot,
                                                     seed=seed)
        rows.append({"label": int(lab), "rows": int(len(g)), "companies": int(g["ticker"].nunique()),
                     "median_move": float(g["move"].median()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def preprocessing_effect(
    raw: pd.DataFrame,
    production: pd.DataFrame,
    columns: Sequence[str],
    tickers: Any,
    dates: Any,
    cluster_opts: Mapping[str, Any],
    seed: int,
    top: float = 0.01,
    reference_labels: Any = None,
) -> dict[str, Any]:
    """StandardScaler on the raw ratios against the production matrix, on the same rows and columns.

    Two numbers per recipe: the share of PC1's sum of squared scores held by the ``top`` most extreme rows
    (correlation PCA; a Gaussian with the same covariance gives about 8%), and the cluster sizes of the
    production clusterer (algorithm, K, unit, window and assignment from ``cluster_opts``) refitted on
    that recipe's matrix. Raw NaN cells (undefined ratios) get the column median before scaling, the same
    fill the production recipe applies after its own scaling. With ``reference_labels`` (the stage-05
    labels) each recipe also reports its ARI with them; the production recipe must give 1.0.
    """
    R = raw[list(columns)].to_numpy(dtype=np.float64)
    med = np.nanmedian(R, axis=0)
    R = np.where(np.isfinite(R), R, med)
    std = R.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    spaces = {"StandardScaler": (R - R.mean(axis=0)) / std,
              "production recipe": production[list(columns)].to_numpy(dtype=np.float64)}
    mask = cluster.fitpredict.window_mask(pd.Series(pd.to_datetime(np.asarray(dates))), cluster_opts["fit_window"])
    out: dict[str, Any] = {"gaussian_reference": float(reduce.pca.gaussian_tail_share(top)), "top_share": top}
    for name, X in spaces.items():
        fit = reduce.pca.fit_pca(X, list(columns), standardize=True)
        tail = float(reduce.pca.score_tail_share(fit.transform(X, 1)[:, 0], top))
        fitted = cluster.fit_unit(
            X, cluster.algorithms.normalize_algorithm(cluster_opts["algorithm"]), int(cluster_opts["k"]),
            dict(cluster_opts.get("params") or {}), seed, unit=cluster_opts["unit"], groups=np.asarray(tickers),
            assign=cluster_opts["assign"], fit_mask=mask, min_rows=int(cluster_opts["min_rows"]),
        )
        sizes = pd.Series(fitted.labels_).value_counts().sort_values(ascending=False)
        out[name] = {
            "pc1_top_share": tail,
            "cluster_rows": [int(v) for v in sizes.to_numpy()],
            "cluster_shares": [float(v) for v in (sizes / sizes.sum()).to_numpy()],
            "pc1_explained": float(fit.explained_ratio[0]),
        }
        if reference_labels is not None:
            out[name]["ari_with_stage05"] = float(adjusted_rand_score(np.asarray(reference_labels), fitted.labels_))
    return out
