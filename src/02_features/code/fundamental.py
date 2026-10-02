"""``fundamental_v2``: a scale-free ratio feature set that fixes the ``baseline12`` feature defects.

Design decisions (formulas and one-line reasons live in :data:`FEATURES`):

- Every feature is a ratio of two quantities in the same currency, so share count and company size
  cancel. Raw ``price``, ``sps`` and ``fcfps`` (in ``baseline12``) are excluded because they move with share
  splits and share count rather than with the business.
- ``pb``, ``sps`` and ``fcfps`` are stored with 3 decimals in fund.csv (of the 25,130 baseline rows,
  56 have ``sps`` = 0.000 and 222 have |sps| < 0.01), so yields are rebuilt from unrounded statement
  items where they exist; ``fcfps`` has no unrounded source and is used as is.
- Valuation enters as yields (``x / price``) rather than multiples (``price / x``): price is always
  positive in this data, so a yield is finite and monotone when earnings, sales or book value cross
  zero, where the multiple jumps from +inf to -inf. ``book_to_market`` is the inverse of ``pb``.
- Leverage uses ``debt_to_assets`` and ``equity_ratio``; both stay defined when equity <= 0
  (1,304 retained rows), unlike debt/equity.
- ``roe`` is computed but set to NaN when equity <= 0 (the sign flips meaning there); the separate
  ``neg_equity`` flag records those rows and is not a clustering feature. ``roe`` is also left out
  of the default clustering list because it is close to ``roa / equity_ratio``.
- Nothing is winsorised or scaled here; heavy tails are the denoise stage's job. Non-finite values
  are stored as NaN and counted in the stage report.

Reporting dimensions in fund.csv differ by column (verified on AAPL 2019-2020 against its filings):
``revenue``, ``opinc``, ``ncfo``, ``sps`` and ``netmargin`` are single-quarter, ``netinccmn`` and ``fcfps``
are trailing twelve months (TTM). ``netmargin`` is quarterly net income over quarterly revenue: the sum of
four consecutive ``netmargin * revenue`` values is within 5% of ``netinccmn`` for 98.9% of the 20,523 rows
where four consecutive quarters exist. Ratios mixing the two dimensions are labelled in their formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

KEYS = ("row_id", "ticker", "calendardate")


@dataclass(frozen=True)
class Feature:
    name: str
    formula: str
    reason: str
    inputs: tuple[str, ...]
    fn: Callable[[pd.DataFrame], pd.Series]
    clustering: bool = True


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise ratio with a zero denominator mapped to NaN instead of +-inf."""
    out = num / den.where(den != 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _positive(den: pd.Series) -> pd.Series:
    """Denominator restricted to > 0; ratios over a non-positive base are undefined, not negative."""
    return den.where(den > 0)


FEATURES: tuple[Feature, ...] = (
    Feature(
        "book_to_market", "equity / (price * sharesbas)  (= 1 / pb from unrounded inputs)",
        "valuation; inverse of pb so it stays finite and monotone when equity crosses zero",
        ("equity", "price", "sharesbas"), lambda d: _div(d["equity"], _positive(d["price"] * d["sharesbas"])),
    ),
    Feature(
        "sales_to_price", "revenue / (price * shareswa)  (= sps / price from unrounded inputs; quarterly sales)",
        "valuation; inverse of P/S (baseline12 'pe'), finite when sales are zero",
        ("revenue", "price", "shareswa"), lambda d: _div(d["revenue"], _positive(d["price"] * d["shareswa"])),
    ),
    Feature(
        "earnings_yield", "netinccmn / (price * shareswa)  (TTM EPS / price)",
        "valuation; inverse of P/E, avoids the P/E blow-up near zero earnings",
        ("netinccmn", "shareswa", "price"), lambda d: _div(d["netinccmn"], _positive(d["price"] * d["shareswa"])),
    ),
    Feature(
        "fcf_yield", "fcfps / price  (TTM free cash flow per share / price)",
        "valuation on cash rather than accrual earnings",
        ("fcfps", "price"), lambda d: _div(d["fcfps"], d["price"]),
    ),
    Feature(
        "netmargin", "netmargin  (raw: quarterly net income / quarterly revenue)",
        "profitability per unit of sales; baseline12 'ros' is the same numerator over quarterly revenue",
        ("netmargin",), lambda d: d["netmargin"].replace([np.inf, -np.inf], np.nan),
    ),
    Feature(
        "operating_margin", "opinc / revenue  (both quarterly; NaN if revenue <= 0)",
        "core profitability before financing and tax, unaffected by one-off items below operating income",
        ("opinc", "revenue"), lambda d: _div(d["opinc"], _positive(d["revenue"])),
    ),
    Feature(
        "roa", "netinccmn / assetsavg  (TTM net income / average assets; NaN if assetsavg <= 0)",
        "return on the whole capital base; no x1000 (scaling is the denoise stage's job)",
        ("netinccmn", "assetsavg"), lambda d: _div(d["netinccmn"], _positive(d["assetsavg"])),
    ),
    Feature(
        "roe", "netinccmn / equity  (TTM; NaN when equity <= 0)",
        "return to shareholders; undefined for negative book equity, and ~ roa / equity_ratio",
        ("netinccmn", "equity"), lambda d: _div(d["netinccmn"], _positive(d["equity"])), clustering=False,
    ),
    Feature(
        "quick", "(cashneq + investmentsc + receivables) / liabilitiesc  (NaN if liabilitiesc <= 0)",
        "short-term liquidity without inventory",
        ("cashneq", "investmentsc", "receivables", "liabilitiesc"),
        lambda d: _div(d["cashneq"] + d["investmentsc"] + d["receivables"], _positive(d["liabilitiesc"])),
    ),
    Feature(
        "current_ratio", "assetsc / liabilitiesc  (NaN if liabilitiesc <= 0)",
        "short-term liquidity including inventory; complements quick",
        ("assetsc", "liabilitiesc"), lambda d: _div(d["assetsc"], _positive(d["liabilitiesc"])),
    ),
    Feature(
        "debt_to_assets", "debt / assets  (NaN if assets <= 0)",
        "leverage that stays defined when equity <= 0, unlike debt/equity",
        ("debt", "assets"), lambda d: _div(d["debt"], _positive(d["assets"])),
    ),
    Feature(
        "equity_ratio", "equity / assets  (NaN if assets <= 0; negative when equity < 0)",
        "solvency; continuous through zero equity, no x1000",
        ("equity", "assets"), lambda d: _div(d["equity"], _positive(d["assets"])),
    ),
    Feature(
        "asset_turnover", "revenue / assetsavg  (quarterly revenue / average assets)",
        "capital intensity: separates asset-light from asset-heavy business models",
        ("revenue", "assetsavg"), lambda d: _div(d["revenue"], _positive(d["assetsavg"])),
    ),
    Feature(
        "ocf_to_assets", "ncfo / assets  (quarterly operating cash flow / assets)",
        "cash profitability, less exposed to accrual accounting than roa",
        ("ncfo", "assets"), lambda d: _div(d["ncfo"], _positive(d["assets"])),
    ),
    Feature(
        "cash_to_assets", "cashneq / assets",
        "cash buffer; high for cash-burning growth and biotech firms",
        ("cashneq", "assets"), lambda d: _div(d["cashneq"], _positive(d["assets"])),
    ),
)

SIZE_FEATURE = Feature(
    "log_mktcap", "log(price * sharesbas)  (NaN if market cap <= 0)",
    "optional size control; off by default so clusters describe business profile, not size",
    ("price", "sharesbas"), lambda d: np.log(_positive(d["price"] * d["sharesbas"])),
)

AUXILIARY = {"neg_equity": "1 if equity <= 0 else 0 (marks rows where roe is NaN; not a clustering feature)"}


def selected(include_size: bool = False) -> tuple[Feature, ...]:
    return FEATURES + ((SIZE_FEATURE,) if include_size else ())


def required_raw(include_size: bool = False) -> tuple[str, ...]:
    return tuple(sorted({c for f in selected(include_size) for c in f.inputs}))


def build_fundamental_v2(panel: pd.DataFrame, include_size: bool = False) -> pd.DataFrame:
    """Return ``row_id, ticker, calendardate``, one column per feature, then ``neg_equity`` (int8)."""
    out = panel[list(KEYS)].copy()
    for feature in selected(include_size):
        out[feature.name] = feature.fn(panel).astype("float64")
    out["neg_equity"] = (panel["equity"] <= 0).astype("int8")
    return out.reset_index(drop=True)
