"""Report figures: one function per PNG, each taking prepared frames and returning a matplotlib figure.

Style comes from ``shared.plotting`` (palette, light surface, dpi 150 on save). Two conventions hold in
every figure: an archetype keeps one colour everywhere (its cluster id indexes the categorical palette,
and ids are ordered by size), and labels use words with spaces, never column names with underscores.
The density backgrounds are grey so the coloured archetype marks stay the only categorical colour.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fundclust import shared

from .naming import display_name

plotting = shared.plotting

# Feature order for archetype tables: profitability, cash generation, balance sheet, valuation, turnover.
FEATURE_ORDER = (
    "operating_margin", "netmargin", "roa", "ocf_to_assets", "earnings_yield", "fcf_yield",
    "equity_ratio", "debt_to_assets", "current_ratio", "quick", "cash_to_assets",
    "book_to_market", "sales_to_price", "asset_turnover",
)
# Ratios read as percentages; the rest (liquidity ratios, book to market, sales to price, turnover) stay
# plain numbers because "88%" for a quick ratio of 0.88 would suggest a share of something.
PERCENT = {
    "operating_margin", "netmargin", "roa", "roe", "ocf_to_assets", "earnings_yield", "fcf_yield",
    "equity_ratio", "debt_to_assets", "cash_to_assets",
}
PRETTY = {
    "operating_margin": "operating margin", "netmargin": "net margin", "roa": "ROA", "roe": "ROE",
    "ocf_to_assets": "op. cash flow / assets", "earnings_yield": "earnings yield", "fcf_yield": "FCF yield",
    "equity_ratio": "equity / assets", "debt_to_assets": "debt / assets", "current_ratio": "current ratio",
    "quick": "quick ratio", "cash_to_assets": "cash / assets", "book_to_market": "book to market",
    "sales_to_price": "sales to price", "asset_turnover": "asset turnover",
}
GREYS = ("#f0efec", "#d9d8d3", "#b4b3ac", "#8a8984", "#5f5e5a")


def _plt():
    plotting.apply_style()
    import matplotlib.pyplot as plt

    return plt


def _grey_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("fc_greys", GREYS)


def pretty(feature: str) -> str:
    return PRETTY.get(feature, feature.replace("_", " "))


def fmt_value(feature: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if feature in PERCENT:
        pct = 100 * value
        return f"{pct:.0f}%" if abs(pct) >= 10 else f"{pct:.1f}%"
    return f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"


def archetype_color(label: int) -> str:
    return plotting.CATEGORICAL[label] if 0 <= label < len(plotting.CATEGORICAL) else plotting.OTHER_GREY


def axis_label(axis: Mapping[str, Any], explained: float | None = None) -> str:
    """``PC1, profitability (35% of variance)``; the named direction is always the positive one."""
    text = axis["pc"] if not axis.get("name") else f"{axis['pc']}, {axis['name']}"
    if axis.get("sign", 1) < 0:
        text += " (sign flipped)"
    if explained is not None:
        text += f" ({explained:.0%} of variance)"
    return text


def _title(fig, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, x=0.01, ha="left", y=0.995, color=plotting.TEXT_PRIMARY, fontsize=11)
    if subtitle:
        fig.text(0.01, 0.945, subtitle, ha="left", va="top", color=plotting.TEXT_SECONDARY, fontsize=8.5)


def axis_profiles_figure(
    profiles: pd.DataFrame,
    axes: Sequence[Mapping[str, Any]],
    top: Mapping[str, Sequence[Mapping[str, Any]]],
    explained: Sequence[float],
    n_top: int = 4,
) -> Any:
    """One text heatmap per named axis: its top features at the 1st..99th percentile of the axis scores.

    For an axis whose name runs against the PC's sign, the columns are mirrored so left is always "less"
    and right "more" of the named quantity, and the percentiles are relabelled on the oriented score.
    """
    plt = _plt()
    n = len(axes)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, grid = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 0.9 + 2.25 * nrow), squeeze=False)
    for idx, axis in enumerate(axes):
        ax = grid[idx // ncol][idx % ncol]
        pc = axis["pc"]
        sub = profiles[profiles["pc"] == pc]
        wide = sub.pivot(index="feature", columns="quantile", values="value")
        qs = sorted(wide.columns)
        pos = sub.drop_duplicates("quantile").set_index("quantile")["sigma"]
        if axis.get("sign", 1) < 0:
            qs = qs[::-1]
        feats = [d["feature"] for d in top[pc][:n_top]]
        vals = wide.loc[feats, qs]
        centre = vals[0.5] if 0.5 in vals.columns else vals.mean(axis=1)
        dev = vals.sub(centre, axis=0)
        span = dev.abs().max(axis=1).replace(0, np.nan)
        colour = dev.div(span, axis=0).fillna(0).to_numpy()
        ax.imshow(colour, cmap=plotting.DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")
        for i, f in enumerate(feats):
            for j, q in enumerate(qs):
                ink = "#ffffff" if abs(colour[i, j]) > 0.7 else plotting.TEXT_PRIMARY
                ax.text(j, i, fmt_value(f, vals.loc[f, q]), ha="center", va="center", fontsize=8, color=ink)
        sign = axis.get("sign", 1)
        # Rounding first stops -0.04 from printing as "-0.0 sd".
        ticks = [f"p{100 * (q if sign > 0 else 1 - q):g}\n{round(sign * pos[q], 1) + 0.0:+.1f} sd" for q in qs]
        ax.set_xticks(range(len(qs)), ticks, fontsize=7.5)
        ax.set_yticks(range(len(feats)), [pretty(f) for f in feats], fontsize=8)
        ax.grid(False)
        k = int(pc[2:]) - 1
        ax.set_title(axis_label(axis, explained[k] if k < len(explained) else None), fontsize=9.5, loc="left")
    for idx in range(n, nrow * ncol):
        grid[idx // ncol][idx % ncol].set_visible(False)
    _title(fig, "What each principal axis means in ratio units",
           "A firm at the given percentile of one axis and at the panel mean on the others; "
           "colour = change from the median firm (blue lower, red higher).")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _clip_limits(v: np.ndarray, clip: float) -> tuple[float, float]:
    lo, hi = np.quantile(v, [clip, 1 - clip])
    pad = 0.04 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _column_positions(y: np.ndarray, min_gap: float, lo: float, hi: float) -> np.ndarray:
    """Label heights for a single column: the points' own heights pushed apart to ``min_gap``, kept in range."""
    order = np.argsort(y)
    pos = np.asarray(y, dtype=float)[order].copy()
    for _ in range(100):
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - min_gap / 2, mid + min_gap / 2
        pos = np.clip(pos, lo, hi)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _label_column(ax, xs, ys, texts, ylim: tuple[float, float], fontsize: float = 7.5) -> None:
    """Labels in one column just right of the panel, joined to their points by thin leader lines.

    Sector medians crowd into a small area, so labels next to the points collide; a column keeps every
    label readable at the cost of a leader line.
    """
    dy = ylim[1] - ylim[0]
    ly = _column_positions(np.asarray(ys, dtype=float), 0.065 * dy, ylim[0] + 0.03 * dy, ylim[1] - 0.03 * dy)
    for x0, y0, yy, text in zip(xs, ys, ly, texts):
        ax.annotate(text, xy=(x0, y0), xycoords="data", xytext=(1.03, yy), textcoords=("axes fraction", "data"),
                    fontsize=fontsize, color=plotting.TEXT_PRIMARY, va="center", ha="left", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": plotting.OTHER_GREY, "lw": 0.6, "shrinkA": 1,
                                "shrinkB": 4})


def pc_map_figure(
    scores: np.ndarray,
    axes: Sequence[Mapping[str, Any]],
    explained: Sequence[float],
    sector_points: pd.DataFrame,
    centroids: pd.DataFrame,
    clip: float = 0.005,
) -> Any:
    """PC1-PC2 density with archetype centroids (left) and a zoom on the sector medians (right).

    ``scores`` are the oriented PC1, PC2 scores of every row. ``sector_points`` has ``sector, x, y,
    companies``; ``centroids`` has ``label, name, x, y``. The sector medians sit within about two units of
    each other while the panel spans more than ten, so they get their own zoomed panel instead of
    overlapping labels on the full map.
    """
    plt = _plt()
    from matplotlib.patches import Rectangle

    x, y = scores[:, 0], scores[:, 1]
    xl, yl = _clip_limits(x, clip), _clip_limits(y, clip)
    fig, (ax, az) = plt.subplots(1, 2, figsize=(14.0, 6.0), gridspec_kw={"width_ratios": [1.45, 1]})
    hb = ax.hexbin(x, y, gridsize=70, extent=(*xl, *yl), bins="log", cmap=_grey_cmap(), mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("ticker-quarters per hexagon (log scale)", fontsize=8)
    for _, row in centroids.iterrows():
        ax.scatter([row["x"]], [row["y"]], s=140, color=archetype_color(int(row["label"])), edgecolor="#ffffff",
                   linewidth=2.0, zorder=5, label=display_name(row["name"]))
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xlabel(axis_label(axes[0], explained[0]))
    ax.set_ylabel(axis_label(axes[1], explained[1]))
    ax.legend(loc="upper left", fontsize=8, title="Archetype centroid", title_fontsize=8.5, alignment="left")
    ax.set_title("All ticker-quarters", loc="left", fontsize=9.5)

    if len(sector_points):
        pts = pd.concat([sector_points[["x", "y"]], centroids[["x", "y"]]])
        inside = pts[(pts["x"] >= sector_points["x"].min() - 1) & (pts["x"] <= sector_points["x"].max() + 1)
                     & (pts["y"] >= sector_points["y"].min() - 1) & (pts["y"] <= sector_points["y"].max() + 1)]
        zx = (inside["x"].min() - 0.3, inside["x"].max() + 0.3)
        zy = (inside["y"].min() - 0.3, inside["y"].max() + 0.3)
        ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False,
                               edgecolor=plotting.TEXT_PRIMARY, lw=0.9, ls="--", zorder=6))
        az.hexbin(x, y, gridsize=30, extent=(*zx, *zy), bins="log", cmap=_grey_cmap(), mincnt=1, linewidths=0,
                  alpha=0.35)
        for _, row in centroids.iterrows():
            if zx[0] <= row["x"] <= zx[1] and zy[0] <= row["y"] <= zy[1]:
                az.scatter([row["x"]], [row["y"]], s=140, color=archetype_color(int(row["label"])),
                           edgecolor="#ffffff", linewidth=2.0, zorder=4)
        az.scatter(sector_points["x"], sector_points["y"], marker="D", s=30, color=plotting.TEXT_PRIMARY,
                   edgecolor="#ffffff", linewidth=0.8, zorder=5, label="sector median company")
        texts = [f"{r['sector']} ({int(r['companies'])})" for _, r in sector_points.iterrows()]
        _label_column(az, sector_points["x"], sector_points["y"], texts, zy)
        az.set_xlim(*zx)
        az.set_ylim(*zy)
        az.legend(loc="lower left", fontsize=8)
        az.set_title("Zoom: median company of each Nasdaq sector (companies)", loc="left", fontsize=9.5)
        az.set_xlabel(axis_label(axes[0]))
        az.set_ylabel(axis_label(axes[1]))
    else:
        az.set_visible(False)
    _title(fig, "The first two principal axes as a map of business models",
           f"Grey: density of all ticker-quarters, view clipped to the {100 * clip:g}th-{100 * (1 - clip):g}th "
           "percentiles. Discs: archetype centroids; the dashed box is the zoom on the right.")
    fig.subplots_adjust(left=0.05, right=0.84, bottom=0.1, top=0.86, wspace=0.3)
    return fig


def trajectories_figure(
    paths: Mapping[str, pd.DataFrame],
    background: np.ndarray,
    axes: Sequence[Mapping[str, Any]],
    highlight: str,
    absent: Sequence[str],
    clip: float = 0.005,
    ncol: int = 5,
) -> Any:
    """Small multiples: each ticker's quarterly path on PC1-PC2 over a grey density of all rows.

    ``paths[ticker]`` has ``quarter`` (label), ``x``, ``y`` sorted by date; ``highlight`` is the quarter label
    drawn in red (e.g. ``2020Q2``).
    """
    plt = _plt()
    tickers = list(paths)
    n = max(len(tickers), 1)
    nrow = int(np.ceil(n / ncol))
    allx = np.concatenate([background[:, 0]] + [p["x"].to_numpy() for p in paths.values()])
    ally = np.concatenate([background[:, 1]] + [p["y"].to_numpy() for p in paths.values()])
    xl = (min(_clip_limits(background[:, 0], clip)[0], np.nanmin(allx)), max(_clip_limits(background[:, 0], clip)[1], np.nanmax(allx)))
    yl = (min(_clip_limits(background[:, 1], clip)[0], np.nanmin(ally)), max(_clip_limits(background[:, 1], clip)[1], np.nanmax(ally)))
    fig, grid = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.45 * nrow + 0.9), sharex=True, sharey=True,
                             squeeze=False)
    from matplotlib.ticker import MaxNLocator

    red = plotting.CATEGORICAL[7]
    blue = plotting.CATEGORICAL[0]
    for idx in range(nrow * ncol):
        ax = grid[idx // ncol][idx % ncol]
        if idx >= len(tickers):
            ax.set_visible(False)
            continue
        t = tickers[idx]
        p = paths[t]
        ax.hexbin(background[:, 0], background[:, 1], gridsize=40, extent=(*xl, *yl), bins="log", cmap=_grey_cmap(),
                  mincnt=1, linewidths=0, alpha=0.55)
        ax.plot(p["x"], p["y"], color=blue, lw=1.3, marker="o", ms=2.6, zorder=3)
        ax.scatter(p["x"].iloc[:1], p["y"].iloc[:1], s=30, facecolor="#ffffff", edgecolor=blue, lw=1.3, zorder=4)
        hit = p[p["quarter"] == highlight]
        if len(hit):
            ax.scatter(hit["x"], hit["y"], s=46, color=red, edgecolor="#ffffff", lw=1.2, zorder=5)
        ax.set_title(f"{t}  ({p['quarter'].iloc[0]} to {p['quarter'].iloc[-1]})", fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(False)
    for c in range(ncol):
        ax = grid[nrow - 1][c] if grid[nrow - 1][c].get_visible() else grid[max(nrow - 2, 0)][c]
        ax.set_xlabel(axis_label(axes[0]), fontsize=8)
    for r in range(nrow):
        grid[r][0].set_ylabel(axis_label(axes[1]), fontsize=8)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=blue, lw=1.3, marker="o", ms=3, label="quarterly path"),
        Line2D([], [], color=blue, lw=0, marker="o", ms=6, markerfacecolor="#ffffff", label="first quarter"),
        Line2D([], [], color=red, lw=0, marker="o", ms=7, label=highlight),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=3, fontsize=8)
    sub = "Tickers fixed in the configuration before plotting."
    if absent:
        sub += " Not in the clustered panel: " + ", ".join(absent) + "."
    _title(fig, "Company paths on the first two principal axes, 2016Q1 to 2020Q3", sub)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def archetype_profiles_figure(
    medians: pd.DataFrame, overall_median: pd.Series, overall_iqr: pd.Series, names: Mapping[int, Mapping[str, Any]],
    companies: Mapping[int, int], clipped: pd.DataFrame | None = None, clip: float = 2.0, clipped_share: float = 0.1,
) -> Any:
    """Heatmap of archetype medians: colour = robust z ``(median - panel median) / panel IQR``, text = value.

    ``clipped`` (archetype x feature share of rows beyond the stage-03 winsor bounds) marks a cell with a
    dagger when stage 03 clipped at least ``clipped_share`` of the group's rows to a bound.
    """
    plt = _plt()
    feats = [f for f in FEATURE_ORDER if f in medians.columns]
    labels = list(medians.index)
    z = (medians[feats] - overall_median[feats]) / overall_iqr[feats].replace(0, np.nan)
    zc = z.clip(-clip, clip).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(13.0, 1.8 + 0.55 * len(labels)))
    im = ax.imshow(zc, cmap=plotting.DIVERGING_CMAP, vmin=-clip, vmax=clip, aspect="auto")
    marked = False
    for i, lab in enumerate(labels):
        for j, f in enumerate(feats):
            ink = "#ffffff" if abs(zc[i, j]) > 0.65 * clip else plotting.TEXT_PRIMARY
            text = fmt_value(f, medians.loc[lab, f])
            if (clipped is not None and f in clipped.columns and lab in clipped.index
                    and clipped.loc[lab, f] >= clipped_share):
                text += " †"
                marked = True
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(feats)), [pretty(f) for f in feats], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)),
                  [f"{display_name(names[int(l)]['name'])}  ({companies.get(int(l), 0)} companies)" for l in labels],
                  fontsize=8.5)
    ax.set_xlim(-0.5, len(feats) - 0.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(f"robust z of the median (clipped at ±{clip:g})", fontsize=8)
    sub = ("Text: the archetype's median over its ticker-quarters. Colour: distance from the panel median in "
           "interquartile ranges (red higher, blue lower).")
    if marked:
        sub += (f"\n† At least {clipped_share:.0%} of the group's rows lie beyond the 1%/99% winsor bounds and were "
                "clipped before clustering, so the value is a rough level (margins below -100% come from revenue "
                "near zero).")
    _title(fig, "Five archetypes: median ratios of each group", sub)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if marked else 0.92))
    return fig


def archetype_sectors_figure(
    mix: Mapping[str, pd.DataFrame], names: Mapping[int, Mapping[str, Any]]
) -> Any:
    """Archetype x sector heatmap: text = share of the archetype's companies, colour = log2 lift."""
    plt = _plt()
    share, lift, counts = mix["share"], mix["lift"], mix["counts"]
    sectors = list(counts.sum(axis=0).sort_values(ascending=False).index)
    share, lift = share[sectors], lift[sectors]
    labels = list(share.index)
    L = np.log2(lift.replace(0, np.nan).to_numpy(dtype=float))
    # A sector with no company in the archetype has lift 0 (log -inf); it is drawn neutral and labelled
    # "none" so empty cells do not read as the strongest signal on the chart.
    empty = counts[sectors].reindex(labels).to_numpy() == 0
    Lc = np.clip(np.nan_to_num(L, nan=0.0), -2, 2)
    Lc[empty] = 0.0
    fig, ax = plt.subplots(figsize=(13.0, 1.9 + 0.6 * len(labels)))
    im = ax.imshow(Lc, cmap=plotting.DIVERGING_CMAP, vmin=-2, vmax=2, aspect="auto")
    for i, lab in enumerate(labels):
        for j, s in enumerate(sectors):
            sh, li = share.loc[lab, s], lift.loc[lab, s]
            if empty[i, j]:
                ax.text(j, i, "none", ha="center", va="center", fontsize=7, color=plotting.TEXT_SECONDARY)
                continue
            ink = "#ffffff" if abs(Lc[i, j]) > 1.3 else plotting.TEXT_PRIMARY
            share_text = f"{100 * sh:.0f}%" if sh >= 0.005 else "<1%"
            ax.text(j, i - 0.13, share_text, ha="center", va="center", fontsize=8, color=ink)
            ax.text(j, i + 0.22, f"×{li:.1f}", ha="center", va="center", fontsize=6.5, color=ink)
    ax.set_xticks(range(len(sectors)), sectors, rotation=30, ha="right", fontsize=8)
    n_known = counts.sum(axis=1)
    unl = mix["unlabelled"]
    ax.set_yticks(range(len(labels)), [
        f"{display_name(names[int(l)]['name'])}  ({int(n_known[l])} with sector, {int(unl.get(l, 0))} without)"
        for l in labels], fontsize=8.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["≤ ×0.25", "×0.5", "×1", "×2", "≥ ×4"])
    cb.set_label("lift: share in archetype / share in panel", fontsize=8)
    _title(fig, "Sector mix of each archetype",
           "Companies in their most frequent archetype. Top number: share of the archetype's companies with a "
           "Nasdaq sector; bottom: lift over the panel; none = no company.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def migrations_figure(mig: Mapping[str, Any], highlight: str, by_sector: pd.DataFrame | None = None) -> Any:
    """Left: per-quarter move rate with a 95% company-bootstrap band; right: the highlighted quarter by sector.

    The sector panel lists every sector with a company at risk and prints ``moved / at risk`` next to each
    bar, because most sectors have a few dozen companies at risk per quarter.
    """
    plt = _plt()
    tab = mig["table"]
    x = np.arange(len(tab))
    if by_sector is not None and len(by_sector):
        fig, (ax, axs) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9.6, 4.6))
        axs = None
    blue = plotting.CATEGORICAL[0]
    red = plotting.CATEGORICAL[7]
    ax.fill_between(x, 100 * tab["ci_low"], 100 * tab["ci_high"], color=plotting.SEQUENTIAL_BLUE[0], lw=0,
                    label="95% CI, company bootstrap")
    ax.plot(x, 100 * tab["rate"], color=blue, marker="o", ms=4, label="share that moved")
    base = 100 * mig["base_rate"]
    ax.axhline(base, color=plotting.TEXT_SECONDARY, lw=1.0, ls="--",
               label=f"{mig['base_quarters'][0]}-{mig['base_quarters'][1]} pooled rate, {base:.1f}%")
    where = np.flatnonzero(tab["quarter"].to_numpy() == highlight)
    if len(where):
        i = int(where[0])
        row = tab.iloc[i]
        ax.scatter([i], [100 * row["rate"]], s=60, color=red, edgecolor="#ffffff", lw=1.5, zorder=5)
        ax.annotate(
            f"{highlight}: {100 * row['rate']:.1f}% ({int(row['moved'])} of {int(row['at_risk'])}),\n"
            f"{row['ratio_to_base']:.2f}× the base rate (95% CI {row['ratio_ci_low']:.2f}-{row['ratio_ci_high']:.2f})",
            (i, 100 * row["rate"]), xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
            fontsize=8, color=plotting.TEXT_PRIMARY)
    ax.set_xticks(x, tab["quarter"], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("% of companies in a profitable archetype\nthe quarter before")
    ax.set_ylim(0, max(np.nanmax(np.r_[100 * tab["ci_high"].to_numpy(dtype=float), base]), 1.0) * 1.15)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("All companies, quarter by quarter", loc="left", fontsize=9.5)
    if axs is not None:
        s = by_sector.sort_values("rate", ascending=True, na_position="first").reset_index(drop=True)
        y = np.arange(len(s))
        axs.barh(y + 0.19, 100 * s["rate"].fillna(0), height=0.36, color=red, label=highlight)
        axs.barh(y - 0.19, 100 * s["base_rate"].fillna(0), height=0.36, color=plotting.OTHER_GREY,
                 label=f"{mig['base_quarters'][0]}-{mig['base_quarters'][1]} pooled")
        top = 100 * np.nanmax(np.r_[s["rate"].to_numpy(dtype=float), s["base_rate"].to_numpy(dtype=float)])
        for k, row in s.iterrows():
            rate = row["rate"] if np.isfinite(row["rate"]) else 0.0
            axs.text(100 * rate + 0.015 * top, k + 0.19, f"{int(row['moved'])}/{int(row['at_risk'])}", va="center",
                     fontsize=7, color=plotting.TEXT_PRIMARY)
        axs.set_yticks(y, s["sector"], fontsize=8)
        axs.set_xlim(0, max(top, 1.0) * 1.25)
        axs.set_xlabel("% moved into a loss-making archetype")
        axs.legend(loc="lower right", fontsize=8)
        axs.set_title(f"{highlight} by Nasdaq sector (moved / at risk)", loc="left", fontsize=9.5)
        axs.grid(axis="y", visible=False)
    _title(fig, "Moves from a profitable into a loss-making archetype",
           "Loss-making archetypes: median ROA below 0. Only consecutive quarters of the same company count.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def risk_figure(risk: pd.DataFrame, names: Mapping[int, Mapping[str, Any]], caps: pd.Series | None = None) -> Any:
    """Median quarter-adjusted next-quarter move per archetype with its ticker-bootstrap 95% CI.

    ``caps`` (median company market cap per archetype, dollars) goes into the row labels: the order of the
    moves partly follows company size, and a reader should see that next to the moves.
    """
    plt = _plt()
    r = risk.sort_values("median_move").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.6, 0.9 + 0.6 * len(r)))
    for i, row in r.iterrows():
        c = archetype_color(int(row["label"]))
        ax.plot([100 * row["ci_low"], 100 * row["ci_high"]], [i, i], color=c, lw=3.0, solid_capstyle="round")
        ax.scatter([100 * row["median_move"]], [i], s=60, color=c, edgecolor="#ffffff", lw=1.5, zorder=3)
        ax.text(100 * row["ci_high"] + 0.4, i,
                f"{100 * row['median_move']:.1f}% ({100 * row['ci_low']:.1f}-{100 * row['ci_high']:.1f})",
                va="center", fontsize=8, color=plotting.TEXT_PRIMARY)

    def _label(lab: int, n: int) -> str:
        text = f"{display_name(names[int(lab)]['name'])}  ({int(n):,} quarters"
        if caps is not None and int(lab) in caps.index:
            text += f", median cap ${caps[int(lab)] / 1e9:.1f}B"
        return text + ")"

    ax.set_yticks(range(len(r)), [_label(l, n) for l, n in zip(r["label"], r["rows"])], fontsize=8.5)
    ax.set_xlim(0, 100 * r["ci_high"].max() * 1.3)
    ax.set_ylim(-0.6, len(r) - 0.4)
    ax.set_xlabel("median size of the next-quarter price move beyond the quarter's market median (log points, %)")
    _title(fig, "Next-quarter price moves by archetype",
           "Dot: median over ticker-quarters; bar: 95% CI from resampling companies. Smaller companies sit in "
           "the riskier archetypes, so part of the order is size.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def preprocessing_figure(effect: Mapping[str, Any]) -> Any:
    """Left: PC1 share held by the top 1% rows per recipe; right: cluster size shares per recipe."""
    plt = _plt()
    recipes = ["StandardScaler", "production recipe"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    vals = [100 * effect[r]["pc1_top_share"] for r in recipes]
    bars = ax1.barh(range(len(recipes)), vals, color=plotting.CATEGORICAL[0], height=0.55)
    for b, v in zip(bars, vals):
        ax1.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5,
                 color=plotting.TEXT_PRIMARY)
    g = 100 * effect["gaussian_reference"]
    ax1.axvline(g, color=plotting.TEXT_SECONDARY, ls="--", lw=1.0)
    ax1.text(g + 1.5, -0.55, f"Gaussian reference {g:.1f}%", fontsize=7.5, color=plotting.TEXT_SECONDARY,
             va="center")
    ax1.set_yticks(range(len(recipes)), [r.capitalize() if r[0].islower() else r for r in recipes])
    ax1.set_xlim(0, 112)
    ax1.set_ylim(-0.8, len(recipes) - 0.5)
    ax1.invert_yaxis()
    ax1.set_xlabel(f"share of PC1's sum of squares from the top {100 * effect['top_share']:g}% rows (%)")
    ax1.set_title("How much of the first axis a few rows own", loc="left", fontsize=9.5)
    ramp = plotting.SEQUENTIAL_BLUE[::-1]
    for i, r in enumerate(recipes):
        left = 0.0
        for k, s in enumerate(effect[r]["cluster_shares"]):
            w = 100 * s
            ax2.barh(i, w, left=left, color=ramp[min(k, len(ramp) - 1)], height=0.55, edgecolor=plotting.SURFACE,
                     linewidth=2)
            if w >= 6:
                ink = "#ffffff" if k < 4 else plotting.TEXT_PRIMARY
                ax2.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=ink)
            left += w
        small = [k for k, s in enumerate(effect[r]["cluster_shares"]) if 100 * s < 6]
        if small:
            rows, shares = effect[r]["cluster_rows"], effect[r]["cluster_shares"]
            text = "\n".join(f"{100 * shares[k]:.2g}% ({rows[k]:,} rows)" for k in small)
            ax2.text(101, i, text, va="center", fontsize=7.5, color=plotting.TEXT_SECONDARY)
    ax2.set_yticks(range(len(recipes)), [r.capitalize() if r[0].islower() else r for r in recipes])
    ax2.set_xlim(0, 140)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.invert_yaxis()
    ax2.set_xlabel("share of ticker-quarters per cluster, largest first (%)")
    ax2.set_title("k-means, K = 5, production unit and fit window", loc="left", fontsize=9.5)
    _title(fig, "Same 14 ratios, two preprocessing recipes",
           "StandardScaler on the raw ratios against winsorizing, asinh and median/IQR scaling (stage 03).")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
