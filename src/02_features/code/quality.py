"""Data-quality flags: rows whose raw statements look wrong, marked for review and never clustered.

The flags are 0/1 ``int8`` columns written next to the features and listed under ``auxiliary`` in
``feature_meta.json``, so ``03_denoise`` (which reads only ``clustering_features``) never sees them. They
mark rows for a reader or a later filter; nothing in the pipeline drops or changes a flagged row.

The three rules come from the outlier review of the 2026-10-01 denoising study, which traced most of the
rows that every outlier detector agreed on to recognisable states and a handful to data problems that
no ratio transform can repair:

    dq_home_currency_dr   depositary receipts whose statements are in the home currency while the price
                          is in dollars (TM and HMC report yen, YNDX roubles), so every price ratio is off
                          by the exchange rate
    dq_share_unit_error   a weighted-average share count that leaves the company's level by a factor of
                          ten or more and comes back (VXRT 2017: about 136 thousand against 3.51 million
                          before and after), which a split or an issuance does not do
    dq_revenue_collapse   quarterly revenue below 1% of the company's own median while total assets stay
                          within half to twice their median (DVN 2017Q4: $3.0 million against a quarterly
                          median of $1.9 billion)

A rule whose raw inputs are absent from the panel is skipped and reported as skipped; a row with a missing
input, or with a share count or market cap that is not positive, cannot be judged and gets 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Ticker median of equity / market cap above which home-currency statements are suspected. Measured on
# the 25,130-row panel: HMC 155.6, TM 52.3, YNDX 14.1; the next ticker is NFH at 8.3 and the 5th at 3.3,
# so 10 separates the three known cases without touching a dollar-reporting company.
DR_BOOK_TO_MARKET = 10.0
# A tenfold change in the share count is a unit error (thousands against units) rather than issuance.
SHARE_JUMP_FACTOR = 10.0
# Longest run of quarters a unit error may span and still be detected. VXRT's spans 4 (2017Q1-Q4); a
# longer run is indistinguishable from two real corporate actions.
SHARE_RUN_MAX = 4
REVENUE_SHARE = 0.01
ASSET_BAND = (0.5, 2.0)

FLAGS: dict[str, dict[str, Any]] = {
    "dq_home_currency_dr": {
        "suspect": "depositary receipt with statements in the home currency",
        "rule": f"ticker median of equity / (price * sharesbas) > {DR_BOOK_TO_MARKET:g}; every row of the ticker",
        "inputs": ("equity", "price", "sharesbas"),
    },
    "dq_share_unit_error": {
        "suspect": "share-count unit error",
        "rule": (
            f"a run of at most {SHARE_RUN_MAX} consecutive rows of a ticker whose shareswa differs by a factor "
            f">= {SHARE_JUMP_FACTOR:g} from both the row before and the row after the run, jumping away "
            "and back in opposite directions"
        ),
        "inputs": ("shareswa",),
    },
    "dq_revenue_collapse": {
        "suspect": "quarterly revenue collapse with a stable balance sheet",
        "rule": (
            f"revenue < {REVENUE_SHARE:g} x the ticker's median revenue (median > 0) while assets are "
            f"within {ASSET_BAND[0]:g}-{ASSET_BAND[1]:g} x the ticker's median assets"
        ),
        "inputs": ("revenue", "assets"),
    },
}
REQUIRED_RAW = tuple(sorted({c for spec in FLAGS.values() for c in spec["inputs"]}))


def _order(panel: pd.DataFrame) -> np.ndarray:
    """Positions that sort the panel by ticker, then date; flags are computed in that order."""
    return np.lexsort((pd.to_datetime(panel["calendardate"]).to_numpy(), panel["ticker"].astype(str).to_numpy()))


def home_currency_dr(panel: pd.DataFrame, threshold: float = DR_BOOK_TO_MARKET) -> np.ndarray:
    """1 for every row of a ticker whose median book-to-market exceeds ``threshold``.

    The ticker median, not the row value, decides: a currency mismatch is permanent, while a single
    quarter of high book-to-market also occurs after a price crash.
    """
    cap = pd.to_numeric(panel["price"], errors="coerce") * pd.to_numeric(panel["sharesbas"], errors="coerce")
    bm = pd.to_numeric(panel["equity"], errors="coerce") / cap.where(cap > 0)
    med = bm.groupby(panel["ticker"].to_numpy()).transform("median")
    return (med > threshold).fillna(False).to_numpy(dtype=np.int8)


def share_unit_error(
    panel: pd.DataFrame, factor: float = SHARE_JUMP_FACTOR, max_run: int = SHARE_RUN_MAX
) -> np.ndarray:
    """1 for rows inside a short run of share counts that jumps away by ``factor`` and comes back.

    A single jump is not flagged: a reverse split or a merger moves the count once and stays. Rows are
    neighbours when they are adjacent in the ticker's date order, so a reporting gap does not hide a jump.
    """
    order = _order(panel)
    tick = panel["ticker"].astype(str).to_numpy()[order]
    shares = pd.to_numeric(panel["shareswa"], errors="coerce").to_numpy(dtype=np.float64)[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.where(shares > 0, np.log10(shares), np.nan)
    step = np.log10(factor)
    flagged = np.zeros(len(order), dtype=bool)
    starts = np.flatnonzero(np.r_[True, tick[1:] != tick[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for lo, hi in zip(starts, ends):
        v = logv[lo:hi]
        d = np.diff(v)
        # jumps[i] = position (within the ticker) of the first row after a jump.
        jumps = [i + 1 for i in range(len(d)) if np.isfinite(d[i]) and abs(d[i]) >= step]
        for a in jumps:
            for b in jumps:
                if b <= a:
                    continue
                if b - a > max_run:
                    break
                if np.sign(d[b - 1]) == -np.sign(d[a - 1]):
                    run = v[a:b]
                    if np.all(np.abs(run - v[a - 1]) >= step) and np.all(np.abs(run - v[b]) >= step):
                        flagged[lo + a : lo + b] = True
                    break
    out = np.zeros(len(order), dtype=np.int8)
    out[order] = flagged
    return out


def revenue_collapse(
    panel: pd.DataFrame, share: float = REVENUE_SHARE, asset_band: tuple[float, float] = ASSET_BAND
) -> np.ndarray:
    """1 where quarterly revenue falls below ``share`` of the ticker median while assets stay in band.

    The asset condition separates a reporting problem from a company that sold or wound down its business,
    whose assets shrink with its revenue. Negative quarterly revenue (a derived fourth quarter that went
    wrong, or hedging losses booked in revenue) also falls below the threshold and is flagged on purpose.
    Real collapses pass the rule too: cruise lines in 2020Q3 had almost no sailings, so the flag marks rows
    to look at, not rows known to be wrong.
    """
    keys = panel["ticker"].to_numpy()
    rev = pd.to_numeric(panel["revenue"], errors="coerce")
    assets = pd.to_numeric(panel["assets"], errors="coerce")
    med_rev = rev.groupby(keys).transform("median")
    med_assets = assets.groupby(keys).transform("median")
    ratio = assets / med_assets.where(med_assets > 0)
    hit = (med_rev > 0) & (rev < share * med_rev) & ratio.between(*asset_band)
    return hit.fillna(False).to_numpy(dtype=np.int8)


RULES = {
    "dq_home_currency_dr": home_currency_dr,
    "dq_share_unit_error": share_unit_error,
    "dq_revenue_collapse": revenue_collapse,
}


def data_quality_flags(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``(flags, report)``: one ``int8`` column per computable rule, aligned to ``panel``'s rows.

    ``report`` holds, per flag, the rule text, the rows and tickers it catches, and the ``skipped`` rules
    with the input columns the panel lacks.
    """
    flags = pd.DataFrame(index=panel.index)
    report: dict[str, Any] = {"flags": {}, "skipped": {}}
    for name, fn in RULES.items():
        missing = [c for c in FLAGS[name]["inputs"] if c not in panel.columns]
        if missing:
            report["skipped"][name] = missing
            continue
        col = fn(panel)
        flags[name] = col
        hit = col.astype(bool)
        tickers = sorted(pd.unique(panel["ticker"].astype(str).to_numpy()[hit]).tolist())
        report["flags"][name] = {
            "suspect": FLAGS[name]["suspect"],
            "rule": FLAGS[name]["rule"],
            "rows": int(hit.sum()),
            "tickers": len(tickers),
            "ticker_list": tickers,
        }
    return flags, report
