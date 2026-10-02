"""The 12-feature matrix (``baseline12``) of the UMAP + DBSCAN baseline in ``baselines/umap_dbscan/``.

The operation order is fixed on purpose: sums are first divided by ``shareswa`` and the ratios are taken
between per-share values. Algebraically the division cancels, but in floating point ``(a/s)/(b/s)`` and
``a/b`` can differ in the last bit, and ``test_feature_set_matches_stored_checksum`` compares a sha256 of
the whole frame, so reordering the arithmetic would break the check and every baseline run built on it.

Properties of the baseline definition (fixed in ``fundamental_v2``): the
x1000 multipliers on ``roa`` and ``equity_rat`` are meaningless after standardisation; ``pe`` is price /
quarterly revenue per share, i.e. price-to-sales; raw ``price`` and ``sps``/``fcfps`` depend on share
count; ``roe`` and ``debt_eq`` flip sign or explode when equity <= 0; ``ros`` (TTM net income / quarterly
revenue) is about 3.5x ``netmargin`` (quarterly net income / quarterly revenue; median ratio 3.55 on the
25,130-row panel, close to the 4 quarters a TTM numerator spans) and largely redundant with it.
"""

from __future__ import annotations

import pandas as pd

KEYS = ("row_id", "ticker", "calendardate")

# Sums scaled to per-share by weighted average shares before any ratio is taken.
PER_SHARE_COLUMNS = (
    "revenue", "debt", "assets", "opinc", "assetsc", "liabilitiesc",
    "equity", "receivables", "investmentsc", "cashneq", "assetsavg", "netinccmn",
)
# Column order of the baseline matrix: raw columns in fund.csv order, then the derived ratios.
COLUMNS = ("pb", "netmargin", "sps", "price", "fcfps", "quick", "roa", "roe", "ros", "pe", "debt_eq", "equity_rat")

FORMULAS = {
    "pb": "pb (raw: market cap / book equity)",
    "netmargin": "netmargin (raw: quarterly net income / quarterly revenue)",
    "sps": "sps (raw: quarterly revenue per share)",
    "price": "price (raw: split-adjusted share price)",
    "fcfps": "fcfps (raw: TTM free cash flow per share)",
    "quick": "(cashneq + investmentsc + receivables) / liabilitiesc   [per-share terms]",
    "roa": "netinccmn / assets * 1000   [TTM net income; x1000 is part of the baseline definition]",
    "roe": "netinccmn / equity",
    "ros": "netinccmn / revenue   [TTM net income over QUARTERLY revenue]",
    "pe": "price / (revenue / shareswa)   [a price-to-quarterly-sales ratio; the baseline keeps the column name pe]",
    "debt_eq": "debt / equity",
    "equity_rat": "equity / assets * 1000   [x1000 is part of the baseline definition]",
}

# Raw inputs the 12 outputs depend on (opinc, assetsc, assetsavg are divided by shareswa but then dropped).
REQUIRED_RAW = (
    "assets", "cashneq", "debt", "equity", "fcfps", "investmentsc", "liabilitiesc", "netinccmn",
    "netmargin", "pb", "price", "receivables", "revenue", "shareswa", "sps",
)


def build_baseline12(panel: pd.DataFrame) -> pd.DataFrame:
    """Return ``row_id, ticker, calendardate`` + the 12 baseline features, in panel row order."""
    dfs = panel.copy()
    for col in PER_SHARE_COLUMNS:
        dfs[col] = dfs[col] / dfs["shareswa"]
    dfs["quick"] = (dfs["cashneq"] + dfs["investmentsc"] + dfs["receivables"]) / dfs["liabilitiesc"]
    dfs["roa"] = (dfs["netinccmn"] / dfs["assets"]) * 1000
    dfs["roe"] = dfs["netinccmn"] / dfs["equity"]
    dfs["ros"] = dfs["netinccmn"] / dfs["revenue"]
    dfs["pe"] = dfs["price"] / dfs["revenue"]
    dfs["debt_eq"] = dfs["debt"] / dfs["equity"]
    dfs["equity_rat"] = (dfs["equity"] / dfs["assets"]) * 1000
    return dfs[list(KEYS) + list(COLUMNS)].reset_index(drop=True)
