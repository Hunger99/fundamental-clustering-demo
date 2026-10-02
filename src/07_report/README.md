# 07_report

Turns one run's artifacts into the figures, tables and numbers a reader needs: what the principal axes mean, where the archetypes and sectors sit on them, how companies moved, how risky each archetype is and what the preprocessing changed. Every number is read or computed from files that stages 01 to 06 wrote into the same run directory, so a report always describes the run it sits in.

## Inputs and outputs

The stage reads, inside `<run>/artifacts/`:

| File | Used for |
| --- | --- |
| `01_load_data/panel.parquet` | prices for the next-quarter move and market caps |
| `02_features/features.parquet`, `feature_meta.json` | archetype medians in ratio units, the raw side of the preprocessing comparison, data-quality counts |
| `03_denoise/X.parquet`, `denoise_meta.json` | the production side of the preprocessing comparison; winsor bounds for the clipped-median marker |
| `04_reduce/pca.json`, `pca_scores.parquet`, `pca_profiles.parquet` | spectrum, loadings, axis profiles, PC scores |
| `05_cluster/labels.parquet`, `cluster_meta.json` | archetype labels, centroids, the clusterer settings refitted in the preprocessing comparison |
| `06_evaluate/panel.json` | the metric panel in `summary.md` |

Sector labels come from the file `evaluate.external` names (Nasdaq screener sectors, current listings). Without it the sector figure is skipped and the map shows no sector medians; the rest still runs.

It writes to `<run>/artifacts/07_report/`:

| File | Content |
| --- | --- |
| `figures/*.png` | the nine figures below, PNG at dpi 150 in the `00_shared` style |
| `summary.md` | the stage-06 panel (gates, ranking scores, context scores), the archetype table, the axis table, headline numbers and the 2020Q2 sector table |
| `report.json` | every number behind the figures and tables, the resolved `report:` settings and the naming evidence |

## Entry points

```powershell
$env:PYTHONPATH = "src"
python -m 07_report.utils.cli --run-dir <run> [--config NAME|PATH]
scripts\07_report\run.ps1 -RunDir <run> [-Config NAME|PATH]
```

Library use: `from fundclust import report`, then `report.naming.name_archetypes(medians)`, `report.naming.name_axes(corr_loadings, r)`, `report.analysis.migration_rates(...)` and the other functions below, or the whole stage as `report.stage.run_report(run_dir, cfg, out_dir)`.

## Modules

| Module | Contents |
| --- | --- |
| `code/naming.py` | `cluster_medians`, `name_archetypes`, `naming_checks`, `name_axes`, `display_name` |
| `code/analysis.py` | `project_centers`, `ticker_medians`, `sector_mix`, `transitions`, `migration_rates`, `migration_by_sector`, `clip_shares`, `market_cap_by_archetype`, `quarter_adjusted_moves`, `risk_by_archetype`, `preprocessing_effect` |
| `code/figures.py` | one function per figure |
| `code/summary.py` | `render` for `summary.md` |
| `code/stage.py` | `DEFAULTS`, `FIGURES` (captions), `run_report` |
| `utils/cli.py` | the stage command line |

## Naming rules

Cluster ids follow cluster size, so a refit can renumber the same business model. The names come from a fixed rule on the medians of the unscaled ratios, applied to every run:

1. A cluster is loss-making when its median ROA is below 0.
2. Of the loss-making clusters, the one with the lowest median asset turnover is "pre-revenue development stage" when that turnover is below 0.05 per quarter; the others are "loss-making cash burners".
3. Of the profitable clusters, the one with the highest median asset turnover is "thin-margin high-turnover", the one with the highest median cash to assets among the rest is "cash-rich profitable", and the others are "mature profitable leveraged".

A name that falls to two clusters gets a numeric suffix. On the production run the deciding medians are ROA -0.081 and -0.401 for the loss-making clusters against 0.048 to 0.102, asset turnover 0.016 for the pre-revenue cluster against 0.144, 0.341 for the high-turnover cluster against 0.129 and 0.198, and cash 17.8% of assets against 4.0%. `naming_checks` then tests the words the rule does not read: the high-turnover group has the thinnest net margin of the profitable groups (3.4%), the leveraged group the highest debt to assets (38%) and the pre-revenue group the highest quick ratio (6.19). All three hold on the production run, and the rule gives the five names of the clustering study to the same five clusters.

The names come from medians alone, and sector mixes inside each group are wide. The mature profitable leveraged group is profitable, indebted and holds little balance-sheet cash; by company count it is mostly Consumer Discretionary (24%) and Industrials (22%), and Utilities make up 17% of it, 2.5 times their share of the panel. `cash_to_assets` counts cash and equivalents only, without marketable securities, so a company that keeps its liquidity in securities (AAPL is one) lands in this group despite large holdings.

Axes are named the same way. Each candidate name has a signature of features and signs: profitability (operating margin, net margin, ROA), balance-sheet strength (equity ratio, minus debt to assets, current ratio), value (book to market, sales to price) and asset turnover. A name's score on a component is the mean signed correlation loading of its signature, and names are matched one to one to the retained components by the largest total absolute score. The position of a component never decides its name. A component whose matched score is below 0.4 stays unnamed, and a negative score flips the component's sign in every report figure so that higher reads as more of the named quantity. Production scores: PC1 profitability 0.78, PC2 balance-sheet strength 0.62, PC3 value 0.72, PC4 asset turnover 0.77; the largest unmatched score is 0.46.

## Figures

Numbers are from the banked production run `outputs/bank/clustering/default--sharadar_sf1_2016q1-2020q3--20261002T070710Z/`.

| Figure | What it shows |
| --- | --- |
| `pca_scree.png` | The correlation-PCA eigenvalues against the 95th percentile of a column-permutation null (stage-04 figure redrawn from `pca.json`). Four components clear it; they hold 75.1% of the variance. |
| `axis_profiles.png` | Each retained axis in ratio units, for a firm at the 1st, 10th, 50th, 90th and 99th percentile of that axis and at the mean on the others. PC1 runs from -4.77 to +2.39 standard deviations, so percentiles replace the lecture's plus or minus 3 sigma; its 99th percentile firm has an operating margin of 35% and ROA of 13%, its 1st percentile firm an operating margin of -500% and a quick ratio of 8.4. |
| `pc_map.png` | Density of all 25,130 ticker-quarters on PC1 and PC2 with the archetype centroids (k-means centres projected through the stage-04 PCA), and a zoom on the median company of each Nasdaq sector. Health Care's median company is lowest on PC1 (-0.91), Technology's highest on balance-sheet strength (+0.61) and Utilities' lowest (-1.29). |
| `trajectories.png` | Quarterly paths of the tickers fixed in `report.trajectory_tickers`, 2020Q2 in red. DAL and CCL drop far along both axes in 2020Q2; ZM starts in 2019Q3. |
| `archetype_profiles.png` | Median ratios of each archetype, coloured by robust z against the panel median. A dagger marks medians of groups where stage 03 clipped at least 10% of the rows to a winsor bound; for the pre-revenue group that covers 10 of the 14 medians, the margins among them (median operating margin -622%, from revenue close to zero). |
| `archetype_sectors.png` | Sector mix of each archetype's companies (each company in its most frequent archetype) with the lift over the panel's mix. Health Care is 92% of the pre-revenue companies that have a sector (lift 5.6); half of the pre-revenue companies have no current Nasdaq listing. |
| `migrations.png` | Per quarter, the share of companies in a profitable archetype one quarter earlier that sit in a loss-making archetype now, with a 2,000-draw company-bootstrap band and the pooled 2016Q2-2019Q4 rate (3.5%). In 2020Q2 the rate was 10.3% (108 of 1,051), 2.92 times the base (95% CI 2.38-3.57). The right panel lists every sector with its counts: Consumer Discretionary 40 of 231 against a 1.9% base, Health Care 14 of 100 against 4.7%, Energy 8 of 25 against 7.1%; Technology, Utilities and Industrials did not rise. |
| `risk_by_archetype.png` | Median of the absolute next-quarter log price change minus that quarter's cross-sectional median, per archetype, with company-bootstrap 95% CIs: 7.4% (7.1-7.6) for mature profitable leveraged up to 20.9% (19.4-22.9) for pre-revenue. Median company market cap falls along roughly the same order ($6.9B, $4.0B, $2.9B, $2.9B, $1.4B), so part of the order is size. |
| `preprocessing_effect.png` | StandardScaler on the 14 raw ratios against the production recipe, on the same rows. The top 1% of rows hold 95.2% of PC1's sum of squares after StandardScaler and 16.0% after the production recipe (Gaussian reference 8.4%). Refitting the production clusterer on the StandardScaler matrix gives clusters of 52.6%, 39.9% and 7.3% of the rows plus two of 20 and 19 rows (ARI 0.27 with the delivered labels); on the production matrix it reproduces the delivered labels exactly. |

Prices are split-adjusted, not dividend-adjusted, and companies that delisted drop out, so the moves are price-change proxies with a survivorship tilt. Sector labels are current listings: 348 of the 1,474 companies have none, and they make up 39% of the companies in the two loss-making archetypes against 19% in the profitable ones.

## Settings (`report:` in the config)

Missing keys fall back to `stage.DEFAULTS`; `report.json` records the resolved values.

| Key | Default | Why |
| --- | --- | --- |
| `trajectory_tickers` | AAPL, MSFT, TSLA, BA, NFLX, AMZN, WMT, PG, JNJ, KO, XOM, DAL, MAR, CCL, ZM | the companies whose paths are drawn; an absent ticker is skipped and named in the figure and in `report.json` |
| `highlight_quarter` | 2020-06-30 | the lockdown quarter drawn in red and broken down by sector |
| `migration_base` | 2016-04-01 to 2019-12-31 | every quarter-to-quarter move before 2020; 2016Q1 has no previous quarter |
| `n_boot` | 2000 | company bootstraps for the migration and risk intervals |
| `map_clip` | 0.005 | density views clipped to the 0.5th-99.5th score percentiles |
| `profile_features` | 4 | features per axis in `axis_profiles.png`, by absolute correlation loading |
| `seed` | null | falls back to `seeds.global` |

## Tests

`src/08_review/tests/test_report_naming.py` (the rule recovers the five names on the production medians, names follow clusters when ids are permuted, suffixes, the pre-revenue bound, axis names from loadings in any order and sign), `test_report_analysis.py` (projection, migration counts on a panel with a reporting gap, sector counts, quarter-adjusted moves, sector lift, strict clip counting, the preprocessing comparison) and `test_report_cli.py` (stages 02 to 07 on a synthetic panel: all nine figures, report numbers that match the run's labels, a run without earlier stages refused).
