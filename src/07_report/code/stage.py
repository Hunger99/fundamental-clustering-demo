"""Stage 07 orchestration: read one run's artifacts, compute the report numbers, write figures and summaries.

:func:`run_report` reads only files earlier stages wrote into the same run directory (plus the sector file
the evaluate stage is configured with), so a report always describes the run it sits in. Settings missing
from the config's ``report:`` section fall back to :data:`DEFAULTS`; the resolved values go into
``report.json``.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fundclust import evaluate, reduce, shared

from . import analysis, figures, naming, summary

DEFAULTS: dict[str, Any] = {
    # Fixed before any path was drawn (the dimensionality-reduction study's list); a ticker absent from the
    # clustered panel is skipped and named in the figure and in report.json.
    "trajectory_tickers": ["AAPL", "MSFT", "TSLA", "BA", "NFLX", "AMZN", "WMT", "PG", "JNJ", "KO", "XOM", "DAL",
                           "MAR", "CCL", "ZM"],
    "highlight_quarter": "2020-06-30",
    # Quarters pooled into the migration base rate; 2016Q1 has no previous quarter, so the first possible
    # move is 2016Q2.
    "migration_base": ["2016-04-01", "2019-12-31"],
    "n_boot": 2000,
    # View of the density plots clipped to these score quantiles; the extreme 0.5% would squeeze the bulk
    # of 25,130 rows into a few hexagons.
    "map_clip": 0.005,
    "profile_features": 4,
    "seed": None,
}
FIGURES = {
    "pca_scree.png": "Eigenvalues of the correlation PCA against the 95th percentile of a column-permutation null; "
                     "the components above the null are kept.",
    "axis_profiles.png": "Each retained axis in ratio units at the 1st, 10th, 50th, 90th and 99th percentile of its "
                         "scores, named by the loading rule.",
    "pc_map.png": "Density of all ticker-quarters on PC1-PC2 with the median company of each sector and the "
                  "archetype centroids.",
    "trajectories.png": "Quarterly paths of the tickers in report.trajectory_tickers on PC1-PC2, 2020Q2 in red.",
    "archetype_profiles.png": "Median ratios of each named archetype, coloured by robust z against the panel.",
    "archetype_sectors.png": "Sector mix of each archetype's companies and the lift over the panel's sector mix.",
    "migrations.png": "Share of companies moving from a profitable into a loss-making archetype per quarter, with a "
                      "company-bootstrap band and the 2016-2019 base rate.",
    "risk_by_archetype.png": "Median size of the next-quarter price move beyond the quarter's market median, per "
                             "archetype, with company-bootstrap CIs.",
    "preprocessing_effect.png": "StandardScaler against the production recipe on the same ratios: PC1 tail share "
                                "and the production clusterer's cluster sizes.",
}
KEYS = ("row_id", "ticker", "calendardate")


def resolve_settings(section: dict[str, Any] | None) -> dict[str, Any]:
    return shared.config.deep_merge(copy.deepcopy(DEFAULTS), section or {})


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"report input missing: {path}; run the earlier stages into this run dir first")
    return shared.io.read_parquet(path)


def _sectors(cfg: dict[str, Any], log: logging.Logger) -> tuple[pd.Series | None, str]:
    """Ticker -> sector from ``evaluate.external`` (the file stage 06 uses), or ``None`` when unavailable."""
    get = shared.config.get
    rel = get(cfg, "evaluate.external.file", None)
    key = get(cfg, "evaluate.external.key", "ticker")
    col = get(cfg, "evaluate.external.col", "sector")
    if not rel:
        return None, "not configured"
    path = Path(rel)
    if not path.is_absolute():
        try:
            path = shared.paths.data_root() / path
        except shared.paths.DataRootError as exc:
            log.warning("sector labels unavailable: %s", exc)
            return None, f"data root unavailable: {exc}"
    if not path.is_file():
        log.warning("sector labels unavailable: %s does not exist", path)
        return None, f"missing: {path}"
    # keep_default_na=False: tickers such as "NA" must not become missing values.
    ext = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return ext.drop_duplicates(key).set_index(key)[col], str(path)


def run_report(run_dir: str | Path, cfg: dict[str, Any], out_dir: str | Path,
               log: logging.Logger | None = None) -> dict[str, Any]:
    """Compute every report number from ``run_dir``'s artifacts; write figures, ``summary.md``, ``report.json``."""
    log = log or logging.getLogger("fundclust.07_report")
    run_dir = Path(run_dir)
    art = run_dir / "artifacts"
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    opts = resolve_settings(shared.config.get(cfg, "report", {}) or {})
    seed = int(opts["seed"] if opts["seed"] is not None else shared.config.get(cfg, "seeds.global", 55))
    n_boot = int(opts["n_boot"])
    get = shared.config.get

    labels_df = _read(art / "05_cluster" / "labels.parquet")
    labels_df["calendardate"] = pd.to_datetime(labels_df["calendardate"])
    cmeta = shared.io.read_json(art / "05_cluster" / "cluster_meta.json")
    feats = _read(art / "02_features" / "features.parquet")
    fmeta = shared.io.read_json(art / "02_features" / "feature_meta.json")
    X = _read(art / "03_denoise" / "X.parquet")
    panel = _read(art / "01_load_data" / "panel.parquet")
    pca_json = shared.io.read_json(art / "04_reduce" / "pca.json")
    scores_df = _read(art / "04_reduce" / "pca_scores.parquet")
    prof = _read(art / "04_reduce" / "pca_profiles.parquet")
    panel06 = shared.io.read_json(art / "06_evaluate" / "panel.json")

    # Everything is aligned on the clustered rows (labels.parquet order).
    base = labels_df[list(KEYS) + ["label"]].reset_index(drop=True)
    rid = base["row_id"].to_numpy()
    feat_al = feats.set_index("row_id").reindex(rid).reset_index()
    cols = list(fmeta["clustering_features"])
    labels = base["label"].to_numpy()
    tickers = base["ticker"].astype(str).to_numpy()
    dates = base["calendardate"].to_numpy()

    # Axes: names from the loading rule, oriented so the named direction is positive. The maps need two
    # axes even when parallel analysis keeps one, so at least PC1 and PC2 are named and drawn.
    r = int(pca_json["r_keep"])
    r_map = max(r, 2)
    corr = pd.DataFrame(pca_json["correlation_loadings"]).T.loc[pca_json["features"]]
    axes = naming.name_axes(corr, r_map)
    signs = np.array([a["sign"] for a in axes], dtype=float)
    explained = list(pca_json["explained_ratio"])
    sc = scores_df.set_index("row_id").reindex(rid)
    S = sc[[f"PC{k + 1}" for k in range(r_map)]].to_numpy(dtype=float) * signs
    log.info("axes: %s", [(a["pc"], a["name"], a["sign"], round(a["score"], 3)) for a in axes])

    # Archetype names from cluster medians in ratio units.
    medians = naming.cluster_medians(feat_al, labels, [c for c in fmeta["features"] if c in feat_al.columns])
    names = naming.name_archetypes(medians)
    checks = naming.naming_checks(medians, names)
    modal = evaluate.modal_label(labels, tickers)
    companies = modal.value_counts().to_dict()
    log.info("archetypes: %s; checks %s", {k: v["name"] for k, v in names.items()}, checks)
    loss_labels = [k for k, v in names.items() if v["role"] == "loss-making"]

    sectors, sector_source = _sectors(cfg, log)

    # a. scree, reusing the stage-04 figure on this run's pca.json.
    fig = reduce.figures.scree_figure(
        pca_json, title=f"PCA spectrum: {pca_json['r_parallel']} components above the permutation null")
    shared.plotting.savefig(fig, fig_dir / "pca_scree.png")

    # b. axis profiles.
    top = pca_json["top_loadings"]
    shared.plotting.savefig(
        figures.axis_profiles_figure(prof, [a for a in axes if a["pc"] in set(prof["pc"])], top, explained,
                                     n_top=int(opts["profile_features"])),
        fig_dir / "axis_profiles.png")

    # c. map with sector medians (one median per company first) and archetype centroids.
    # Stage 05 stores the centres in cluster-id order (ids ranked by size), so row k is archetype k.
    centers = np.asarray(cmeta["centers"], dtype=float)
    cent = analysis.project_centers(centers, pca_json, cmeta["space_columns"], r_map) * signs
    centroids = pd.DataFrame({"label": sorted(names), "name": [names[k]["name"] for k in sorted(names)],
                              "x": cent[sorted(names), 0], "y": cent[sorted(names), 1]})
    sector_points = pd.DataFrame(columns=["sector", "x", "y", "companies"])
    if sectors is not None:
        tm = analysis.ticker_medians(pd.DataFrame({"ticker": tickers, "x": S[:, 0], "y": S[:, 1]}), ["x", "y"])
        tm["sector"] = sectors.reindex(tm.index).to_numpy(dtype=object)
        tm = tm[tm["sector"].notna()]
        sector_points = (tm.groupby("sector").agg(x=("x", "median"), y=("y", "median"), companies=("x", "size"))
                         .reset_index())
    shared.plotting.savefig(
        figures.pc_map_figure(S[:, :2], axes, explained, sector_points, centroids, clip=float(opts["map_clip"])),
        fig_dir / "pc_map.png")

    # d. trajectories for the tickers listed in report.trajectory_tickers.
    highlight = analysis.quarter_label(int(evaluate.quarter_index([opts["highlight_quarter"]])[0]))
    traj = pd.DataFrame({"ticker": tickers, "date": dates, "x": S[:, 0], "y": S[:, 1]})
    paths, absent = {}, []
    for t in opts["trajectory_tickers"]:
        p = traj[traj["ticker"] == t].sort_values("date")
        if p.empty:
            absent.append(t)
            continue
        q = evaluate.quarter_index(p["date"].to_numpy())
        paths[t] = p.assign(quarter=[analysis.quarter_label(int(v)) for v in q]).reset_index(drop=True)
    if absent:
        log.warning("trajectory tickers not in the clustered panel: %s", absent)
    shared.plotting.savefig(
        figures.trajectories_figure(paths, S[:, :2], axes, highlight, absent, clip=float(opts["map_clip"])),
        fig_dir / "trajectories.png")

    # e. archetype profile heatmap; cells whose group stage 03 clipped to its winsor bounds are marked.
    feat_cols = [c for c in fmeta["features"] if c in feat_al.columns]
    overall_med = feat_al[feat_cols].median()
    overall_iqr = feat_al[feat_cols].quantile(0.75) - feat_al[feat_cols].quantile(0.25)
    clipped = None
    dmeta_path = art / "03_denoise" / "denoise_meta.json"
    if dmeta_path.is_file():
        fitted = shared.io.read_json(dmeta_path).get("fitted") or {}
        win = fitted.get("winsor")
        if win:
            lo = dict(zip(fitted["features"], win["lo"]))
            hi = dict(zip(fitted["features"], win["hi"]))
            clipped = analysis.clip_shares(feat_al, labels, lo, hi)
    shared.plotting.savefig(
        figures.archetype_profiles_figure(medians, overall_med, overall_iqr, names, companies, clipped=clipped),
        fig_dir / "archetype_profiles.png")

    # f. sector mix.
    mix = None
    if sectors is not None:
        mix = analysis.sector_mix(modal, sectors)
        shared.plotting.savefig(figures.archetype_sectors_figure(mix, names), fig_dir / "archetype_sectors.png")
    else:
        log.warning("archetype_sectors.png not drawn: no sector labels (%s)", sector_source)

    # g. migrations, overall and by sector in the highlighted quarter.
    bq = evaluate.quarter_index(opts["migration_base"])
    base_q = (int(bq[0]), int(bq[1]))
    mig = analysis.migration_rates(labels, tickers, dates, loss_labels, base_q, n_boot=n_boot, seed=seed)
    hq = int(evaluate.quarter_index([opts["highlight_quarter"]])[0])
    by_sector = None
    if sectors is not None:
        by_sector = analysis.migration_by_sector(labels, tickers, dates, loss_labels, sectors, hq, base_q)
    shared.plotting.savefig(figures.migrations_figure(mig, highlight, by_sector), fig_dir / "migrations.png")

    # h. risk by archetype on the quarter-adjusted next-quarter move; market cap shown beside it.
    fwd_all = evaluate.forward_returns(panel)
    fwd = pd.Series(fwd_all.to_numpy(), index=panel["row_id"].to_numpy()).reindex(rid).to_numpy()
    moves = analysis.quarter_adjusted_moves(fwd, dates)
    risk = analysis.risk_by_archetype(labels, tickers, moves, n_boot=n_boot, seed=seed)
    pan = panel.set_index("row_id").reindex(rid)
    caps = analysis.market_cap_by_archetype(labels, tickers, (pan["price"] * pan["sharesbas"]).to_numpy())
    risk["median_market_cap"] = risk["label"].map(caps)
    shared.plotting.savefig(figures.risk_figure(risk, names, caps), fig_dir / "risk_by_archetype.png")

    # i. preprocessing effect on the same features and rows.
    x_al = X.set_index("row_id").reindex(rid).reset_index()
    copts = {"algorithm": cmeta["algorithm"], "k": cmeta["K"], "params": cmeta.get("params") or {},
             "unit": cmeta["unit"], "assign": cmeta["assign"], "fit_window": cmeta.get("fit_window"),
             "min_rows": cmeta.get("min_rows", 1)}
    effect = analysis.preprocessing_effect(feat_al, x_al, cols, tickers, dates, copts, int(cmeta["seed"]),
                                           reference_labels=labels)
    shared.plotting.savefig(figures.preprocessing_figure(effect), fig_dir / "preprocessing_effect.png")
    log.info("preprocessing effect: %s", {k: v for k, v in effect.items() if isinstance(v, dict)})

    dq_flags = (fmeta.get("data_quality") or {}).get("flags", {})
    report = {
        "run_dir": str(run_dir.resolve()),
        "run_name": run_dir.name,
        "settings": opts,
        "seed": seed,
        # Relative to this stage's folder, so the record stays true when a verified run moves to the bank.
        "figures": {name: f"figures/{name}" for name in FIGURES if (fig_dir / name).is_file()},
        "figure_captions": FIGURES,
        "panel_06": panel06,
        "axes": axes,
        "explained_ratio": explained[:r_map],
        "r_keep": r,
        "score_range_sigma": pca_json.get("score_range_sigma"),
        "archetypes": {
            int(k): {
                **names[k],
                "companies_modal": int(companies.get(k, 0)),
                "rows": int((labels == k).sum()),
                "median_market_cap": float(caps.get(k, float("nan"))),
                "medians": {c: float(medians.loc[k, c]) for c in medians.columns},
                "share_rows_at_winsor_bound": None if clipped is None or k not in clipped.index else {
                    c: float(clipped.loc[k, c]) for c in clipped.columns},
            }
            for k in sorted(names)
        },
        "naming_checks": checks,
        "centroids_pc": centroids.to_dict(orient="records"),
        "sector_source": sector_source,
        "sector_points": sector_points.to_dict(orient="records"),
        "sector_mix": None if mix is None else {
            "counts": mix["counts"].to_dict(orient="index"),
            "lift": mix["lift"].to_dict(orient="index"),
            "unlabelled": {int(k): int(v) for k, v in mix["unlabelled"].items()},
        },
        "trajectories": {"tickers": list(paths), "absent": absent, "highlight": highlight},
        "migrations": {**{k: v for k, v in mig.items() if k != "table"},
                       "loss_making_labels": loss_labels, "by_quarter": mig["table"].to_dict(orient="records"),
                       "by_sector_highlight": None if by_sector is None else by_sector.to_dict(orient="records")},
        "risk": risk.assign(name=[names[int(l)]["name"] for l in risk["label"]]).to_dict(orient="records"),
        "preprocessing_effect": effect,
        "data_quality": {k: {**{kk: vv for kk, vv in v.items() if kk != "ticker_list"},
                             "tickers_listed": v["ticker_list"][:20]} for k, v in dq_flags.items()},
    }
    shared.io.write_json(report, out / "report.json")
    (out / "summary.md").write_text(summary.render(report, get(cfg, "_meta.name", "default")), encoding="utf-8")
    return report
