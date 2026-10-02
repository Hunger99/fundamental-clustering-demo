"""Load the SHARADAR SF1 quarterly extract and apply the missing-data policy.

``row_id`` is the 0-based data-row position in ``fund.csv``. Every stage and the baseline in
``baselines/umap_dbscan/`` carry it, so their artifacts join on it directly.

Missing-data policies (``missing.policy``):

    all_columns      drop a row with a NaN in ANY column (also the baseline's policy; 25,130 rows)
    feature_columns  drop a row only when a raw column needed by the selected feature set is NaN
    none             keep every row

Reporting dimensions differ between columns (measured on AAPL, see ``src/02_features/README.md``):
``revenue``, ``opinc``, ``ncfo``, ``sps`` and ``netmargin`` are single-quarter values, ``netinccmn`` and
``fcfps`` are trailing twelve months, balance-sheet items are period-end levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ID_COLUMNS = ("ticker", "calendardate", "datekey")
NUMERIC_COLUMNS = (
    "revenue", "pb", "debt", "assets", "netmargin", "ncfo", "shareswa", "sps", "opinc", "assetsc",
    "liabilitiesc", "price", "sharesbas", "equity", "receivables", "investmentsc", "cashneq",
    "assetsavg", "netinccmn", "fcfps",
)
MISSING_POLICIES = ("all_columns", "feature_columns", "none")
# Columns that only firms with a classified (current / non-current) balance sheet report.
CLASSIFIED_BALANCE_SHEET = ("assetsc", "liabilitiesc", "investmentsc")


def read_fund(path: str | Path) -> pd.DataFrame:
    """Read ``fund.csv``: ``row_id`` first, ``ticker`` as str, both date columns as datetime64."""
    df = pd.read_csv(path)
    missing = set(ID_COLUMNS) | set(NUMERIC_COLUMNS)
    missing -= set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks expected columns: {sorted(missing)}")
    df.insert(0, "row_id", np.arange(len(df), dtype="int64"))
    df["ticker"] = df["ticker"].astype(str)
    df["calendardate"] = pd.to_datetime(df["calendardate"], format="%Y-%m-%d")
    df["datekey"] = pd.to_datetime(df["datekey"], format="%Y-%m-%d")
    return df


def read_indicator_definitions(path: str | Path, columns: Iterable[str]) -> pd.DataFrame:
    """Indicator title/description/unit for the given columns, one row per distinct definition."""
    defs = pd.read_csv(path)
    keep = [c for c in ("indicator", "title", "description", "unittype") if c in defs.columns]
    defs = defs[keep]
    defs = defs[defs["indicator"].isin(list(columns))].drop_duplicates()
    return defs.reset_index(drop=True)


def drop_calendardates(df: pd.DataFrame, dates: Sequence[str]) -> pd.DataFrame:
    stamps = pd.to_datetime(list(dates))
    return df[~df["calendardate"].isin(stamps)]


def missing_mask(df: pd.DataFrame, policy: str, required_columns: Sequence[str] | None = None) -> pd.Series:
    """Boolean Series, True for rows the policy drops."""
    if policy == "all_columns":
        return df.drop(columns=["row_id"]).isna().any(axis=1)
    if policy == "feature_columns":
        if not required_columns:
            raise ValueError("policy 'feature_columns' needs the feature set's required raw columns")
        unknown = set(required_columns) - set(df.columns)
        if unknown:
            raise ValueError(f"required columns not in panel: {sorted(unknown)}")
        return df[list(required_columns)].isna().any(axis=1)
    if policy == "none":
        return pd.Series(False, index=df.index)
    raise ValueError(f"unknown missing policy {policy!r}; expected one of {MISSING_POLICIES}")


def _ticker_profile(df: pd.DataFrame, tickers: Iterable[str]) -> dict[str, float | int | None]:
    """Median size and leverage of a ticker group (each ticker's own median first, to weight tickers equally)."""
    sub = df[df["ticker"].isin(list(tickers))]
    if sub.empty:
        return {"n_tickers": 0, "median_assets_usd_bn": None, "median_equity_to_assets": None}
    per = sub.groupby("ticker")[["assets", "equity"]].median()
    ratio = (per["equity"] / per["assets"]).replace([np.inf, -np.inf], np.nan)
    return {
        "n_tickers": int(len(per)),
        "median_assets_usd_bn": float(per["assets"].median() / 1e9),
        "median_equity_to_assets": float(ratio.median()),
    }


def missing_report(
    raw: pd.DataFrame,
    dated: pd.DataFrame,
    drop: pd.Series,
    *,
    policy: str,
    required_columns: Sequence[str] | None,
    dropped_dates: Sequence[str],
    probe_tickers: Sequence[str],
) -> dict:
    """Characterise what the date filter and the missing policy removed.

    Includes a test of the hypothesis that the NaN filter mostly removes financial firms: banks,
    insurers and REITs do not report a classified balance sheet, so ``assetsc``/``liabilitiesc`` are
    NaN in every quarter. The report lists known financial tickers, counts tickers whose classified
    balance-sheet columns are always NaN, and compares their size and equity/assets with kept tickers
    (banks typically run at ~10% equity/assets against ~40-50% for industrial firms).
    """
    kept = dated[~drop]
    dropped = dated[drop]
    tickers_all = set(dated["ticker"])
    tickers_kept = set(kept["ticker"])
    fully_dropped = sorted(tickers_all - tickers_kept)
    partially = sorted(set(dropped["ticker"]) & tickers_kept)

    nan_cols = [c for c in dated.columns if c != "row_id"]
    patterns = (
        dropped[nan_cols].isna().apply(lambda r: "+".join(c for c, v in r.items() if v) or "(none)", axis=1)
        if len(dropped)
        else pd.Series(dtype=str)
    )

    always_unclassified = (
        dated.assign(_u=dated[list(CLASSIFIED_BALANCE_SHEET[:2])].isna().all(axis=1))
        .groupby("ticker")["_u"]
        .all()
    )
    unclassified = set(always_unclassified[always_unclassified].index)

    probes = []
    for ticker in probe_tickers:
        rows = dated[dated["ticker"] == ticker]
        probes.append(
            {
                "ticker": ticker,
                "in_file": bool((raw["ticker"] == ticker).any()),
                "rows": int(len(rows)),
                "rows_kept": int((kept["ticker"] == ticker).sum()),
                "assetsc_nan_rows": int(rows["assetsc"].isna().sum()),
                "liabilitiesc_nan_rows": int(rows["liabilitiesc"].isna().sum()),
                "dropped_entirely": bool(len(rows) > 0 and ticker not in tickers_kept),
            }
        )
    present = [p for p in probes if p["rows"] > 0]

    return {
        "policy": policy,
        "required_columns": list(required_columns) if required_columns else None,
        "rows": {
            "in_file": int(len(raw)),
            "dropped_by_date": int(len(raw) - len(dated)),
            "dropped_dates": list(dropped_dates),
            "after_date_filter": int(len(dated)),
            "dropped_by_missing_policy": int(drop.sum()),
            "kept": int(len(kept)),
        },
        "tickers": {
            "in_file": int(raw["ticker"].nunique()),
            "after_date_filter": int(len(tickers_all)),
            "kept": int(len(tickers_kept)),
            "dropped_entirely": int(len(fully_dropped)),
            "partially_dropped": int(len(partially)),
            "quarters_per_kept_ticker_median": float(kept.groupby("ticker").size().median()) if len(kept) else None,
        },
        "nan_counts_in_dropped_rows": {c: int(v) for c, v in dropped[nan_cols].isna().sum().items() if v},
        "nan_counts_after_date_filter": {c: int(v) for c, v in dated[nan_cols].isna().sum().items() if v},
        "top_nan_patterns_in_dropped_rows": {k: int(v) for k, v in patterns.value_counts().head(10).items()},
        "dropped_rows_by_quarter": {
            d.strftime("%Y-%m-%d"): int(v) for d, v in dropped.groupby("calendardate").size().items()
        },
        "financial_firm_hypothesis": {
            "unclassified_balance_sheet_tickers": int(len(unclassified)),
            "unclassified_among_dropped_entirely": int(len(unclassified & set(fully_dropped))),
            "share_of_dropped_entirely_explained": (
                float(len(unclassified & set(fully_dropped)) / len(fully_dropped)) if fully_dropped else None
            ),
            "profile_kept_tickers": _ticker_profile(dated, tickers_kept),
            "profile_dropped_unclassified_tickers": _ticker_profile(dated, unclassified & set(fully_dropped)),
            "profile_dropped_other_tickers": _ticker_profile(dated, set(fully_dropped) - unclassified),
            "probe_tickers": probes,
            "probe_summary": {
                "probed": len(probes),
                "present_in_file": len(present),
                "dropped_entirely": sum(p["dropped_entirely"] for p in present),
                "kept": sum(not p["dropped_entirely"] for p in present),
                "kept_tickers": [p["ticker"] for p in present if not p["dropped_entirely"]],
                "absent_from_file": [p["ticker"] for p in probes if not p["in_file"]],
            },
        },
        "dropped_entirely_tickers": fully_dropped,
    }


def load_panel(
    fund_path: str | Path,
    *,
    drop_dates: Sequence[str],
    policy: str,
    required_columns: Sequence[str] | None = None,
    probe_tickers: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """Return ``(panel, report)``: the filtered panel in file order and the missing-data report."""
    raw = read_fund(fund_path)
    dated = drop_calendardates(raw, drop_dates)
    drop = missing_mask(dated, policy, required_columns)
    report = missing_report(
        raw,
        dated,
        drop,
        policy=policy,
        required_columns=required_columns,
        dropped_dates=drop_dates,
        probe_tickers=probe_tickers,
    )
    panel = dated[~drop].reset_index(drop=True)
    return panel, report
