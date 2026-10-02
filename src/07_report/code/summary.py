"""Render ``summary.md`` from the ``report.json`` dict: every number in it is read from the run's artifacts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .figures import fmt_value, pretty
from .naming import display_name

GATE_TEXT = {
    "stability_ari": "Stability, mean ARI of refits on 80% of companies",
    "ps_weighted": "Prediction strength, size-weighted",
    "seed_ari": "Seed stability, mean ARI across seeds",
    "noise_frac": "Noise share",
    "max_cluster_share": "Largest cluster share",
}
RANK_ROWS = (
    ("econ_sector_excess", "Economic separation beyond sector (mean of the two excesses below)"),
    ("econ_abs_excess_sector", "Excess epsilon squared of next-quarter abs. move beyond sector"),
    ("econ_vol_excess_sector", "Excess epsilon squared of volatility beyond sector"),
    ("nmi_ticker_excess", "Sector NMI excess over the company permutation null"),
    ("nmi_ticker_z", "Sector NMI z-score"),
)
CONTEXT_ROWS = (
    ("silhouette", "Silhouette in the fitted space"),
    ("persistence_kappa", "Quarter-to-quarter persistence, Cohen's kappa"),
    ("learn_group_acc", "Learnability, GroupKFold accuracy"),
    ("learn_gap", "Learnability leak gap (row split minus grouped)"),
    ("fe_abs_partial_r2", "Partial R squared of abs. move beyond sector and quarter effects"),
    ("ps_min", "Prediction strength, weakest cluster"),
)
CHECK_TEXT = {
    "high_turnover_has_lowest_profitable_netmargin": "The high-turnover group has the thinnest net margin of the "
                                                      "profitable groups",
    "leveraged_has_highest_profitable_debt_to_assets": "The leveraged group has the highest debt to assets of the "
                                                       "profitable groups",
    "pre_revenue_has_highest_quick_ratio": "The pre-revenue group has the highest quick ratio of all groups",
}
KEY_MEDIANS = ("operating_margin", "roa", "asset_turnover", "cash_to_assets", "debt_to_assets", "quick")


def _num(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):.{digits}f}"


def _pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"


def panel_table(panel: Mapping[str, Any]) -> list[str]:
    rule = panel.get("rule", {})
    gates = rule.get("gates", {})
    optional = set(rule.get("optional_gates", ()))
    lines = ["| Role | Score | Value | Rule | Result |", "| --- | --- | --- | --- | --- |"]
    for key, (op, thr) in gates.items():
        v = panel.get(key)
        missing = v is None or (isinstance(v, float) and not np.isfinite(v))
        if missing:
            result = "skipped (not run)" if key in optional else "fail (missing)"
        else:
            ok = v >= thr if op == ">=" else v <= thr
            result = "pass" if ok else "fail"
        lines.append(f"| Gate | {GATE_TEXT.get(key, key)} | {_num(v)} | {op} {thr:g} | {result} |")
    for key, text in RANK_ROWS:
        lines.append(f"| Ranking | {text} | {_num(panel.get(key), 2 if key.endswith('_z') else 3)} | higher is better | |")
    for key, text in CONTEXT_ROWS:
        lines.append(f"| Context | {text} | {_num(panel.get(key))} | reported only | |")
    return lines


def archetype_table(report: Mapping[str, Any], clipped_share: float = 0.1) -> tuple[list[str], bool]:
    """Markdown rows of the archetype table and whether any median carries the winsor-bound marker."""
    risk = {int(r["label"]): r for r in report["risk"]}
    head = ["Id", "Archetype", "Companies", "Ticker-quarters"] + [display_name(pretty(c)) for c in KEY_MEDIANS] + [
        "Median market cap", "Median next-quarter move (95% CI)"]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    marked = False
    for k, a in report["archetypes"].items():
        r = risk.get(int(k))
        move = "n/a" if r is None else f"{_pct(r['median_move'])} ({_pct(r['ci_low'])}-{_pct(r['ci_high'])})"
        at_bound = a.get("share_rows_at_winsor_bound") or {}
        meds = []
        for c in KEY_MEDIANS:
            text = fmt_value(c, a["medians"].get(c, float("nan")))
            if at_bound.get(c, 0.0) >= clipped_share:
                text += " (†)"
                marked = True
            meds.append(text)
        cap = a.get("median_market_cap")
        cap_text = "n/a" if cap is None or not np.isfinite(cap) else f"${cap / 1e9:.1f}B"
        lines.append("| " + " | ".join([str(k), display_name(a["name"]), f"{a['companies_modal']:,}", f"{a['rows']:,}",
                                         *meds, cap_text, move]) + " |")
    return lines, marked


def sector_migration_table(rows: list[Mapping[str, Any]], quarter: str, base: list[str]) -> list[str]:
    lines = [f"| Sector | {quarter}: moved / at risk | {quarter} rate | {base[0]}-{base[1]}: moved / at risk | Base rate |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        rate = "n/a" if r["rate"] is None or not np.isfinite(r["rate"]) else _pct(r["rate"])
        brate = "n/a" if r["base_rate"] is None or not np.isfinite(r["base_rate"]) else _pct(r["base_rate"])
        lines.append(f"| {r['sector']} | {r['moved']} / {r['at_risk']} | {rate} | {r['base_moved']} / "
                     f"{r['base_at_risk']} | {brate} |")
    return lines


def render(report: Mapping[str, Any], config_name: str) -> str:
    p = report["panel_06"]
    failed = p.get("failed") or ""
    verdict = "passes every gate" if p.get("passes") and not failed else f"fails: {failed}"
    lines = [
        f"# Run report: {report['run_name']}",
        "",
        f"Configuration `{config_name}`. Every number below is read from this run's artifacts; "
        "`report.json` holds the full set.",
        "",
        "## Stage-06 metric panel",
        "",
        f"The clustering {verdict} of `evaluate.DEFAULT_RULE`.",
        "",
        *panel_table(p),
        "",
        "## Archetypes",
        "",
        "Names come from the rule in `src/07_report/code/naming.py` applied to the cluster medians. Companies are "
        "counted in their most frequent archetype; medians are over each archetype's ticker-quarters. The move is "
        "the median of |next-quarter log price change minus that quarter's cross-sectional median|, with a 95% CI "
        "from resampling companies.",
        "",
    ]
    table, marked = archetype_table(report)
    lines += table + [""]
    if marked:
        lines += ["(†) At least 10% of the archetype's rows lie beyond the 1%/99% winsor bounds and were clipped "
                  "before clustering, so the median is a rough level; margins below -100% come from revenue close "
                  "to zero.", ""]
    lines += ["Median market cap is the median over companies of each company's median price x shares in the "
              "archetype's quarters. The riskier archetypes hold smaller companies, so part of the order of the "
              "moves is size.", ""]
    checks = report.get("naming_checks") or {}
    if checks:
        lines += ["Checks of the words in the names that the rule does not read:", ""]
        for key, ok in checks.items():
            verdict = "not applicable" if ok is None else ("holds" if ok else "does not hold")
            lines.append(f"- {CHECK_TEXT.get(key, key)}: {verdict}.")
        lines.append("")

    lines += ["## Principal axes", "", "| Axis | Name | Share of variance | Rule score | Range of scores (sd) |",
              "| --- | --- | --- | --- | --- |"]
    rng = report.get("score_range_sigma") or {}
    for a in report["axes"]:
        k = int(a["pc"][2:]) - 1
        ev = report["explained_ratio"][k] if k < len(report["explained_ratio"]) else float("nan")
        span = rng.get(a["pc"])
        span_text = "n/a" if not span else f"{span['min_sigma']:+.2f} to {span['max_sigma']:+.2f}"
        flip = " (sign flipped)" if a.get("sign", 1) < 0 else ""
        lines.append(f"| {a['pc']} | {a['name'] or 'unnamed'}{flip} | {_pct(ev)} | {_num(a['score'], 2)} | {span_text} |")
    lines.append("")

    mig = report["migrations"]
    hl = report["trajectories"]["highlight"]
    row = next((r for r in mig["by_quarter"] if r["quarter"] == hl), None)
    lines += ["## Headline numbers", ""]
    if row is not None:
        lines.append(
            f"- Moves from a profitable into a loss-making archetype: {_pct(mig['base_rate'])} per quarter pooled over "
            f"{mig['base_quarters'][0]}-{mig['base_quarters'][1]} (95% CI {_pct(mig['base_ci'][0])}-{_pct(mig['base_ci'][1])}); "
            f"{hl} {_pct(row['rate'])} ({row['moved']} of {row['at_risk']}), {row['ratio_to_base']:.2f} times the base "
            f"rate (95% CI {row['ratio_ci_low']:.2f}-{row['ratio_ci_high']:.2f}).")
    by_sector = mig.get("by_sector_highlight")
    eff = report["preprocessing_effect"]
    std, prod = eff["StandardScaler"], eff["production recipe"]
    lines.append(
        f"- Top 1% of rows hold {_pct(std['pc1_top_share'])} of PC1's sum of squares after StandardScaler and "
        f"{_pct(prod['pc1_top_share'])} after the production recipe (Gaussian reference "
        f"{_pct(eff['gaussian_reference'])}). The production clusterer refitted on the StandardScaler matrix puts "
        f"{_pct(std['cluster_shares'][0])} of the rows in its largest cluster, against {_pct(prod['cluster_shares'][0])} "
        f"(ARI with the delivered labels {std.get('ari_with_stage05', float('nan')):.3f} and "
        f"{prod.get('ari_with_stage05', float('nan')):.3f}).")
    absent = report["trajectories"]["absent"]
    if absent:
        lines.append(f"- Trajectory tickers not in the clustered panel: {', '.join(absent)}.")
    dq = report.get("data_quality") or {}
    if dq:
        parts = [f"`{k}` {v['rows']} rows / {v['tickers']} ticker{'s' if v['tickers'] != 1 else ''}"
                 for k, v in dq.items()]
        lines.append("- Data-quality flags from stage 02 (not clustering features): " + "; ".join(parts) + ".")
    if by_sector:
        lines += ["", f"### {hl} moves by Nasdaq sector", "",
                  "Every sector with a company at risk; the counts are small, so read the rates with their denominators.",
                  "", *sector_migration_table(by_sector, hl, mig["base_quarters"])]
    lines += ["", "## Figures", ""]
    for name, caption in report["figure_captions"].items():
        if name in report["figures"]:
            lines.append(f"- `figures/{name}`: {caption}")
    return "\n".join(lines) + "\n"
