# 02_features

Builds a named feature set from the stage-01 panel. Values are unscaled and not winsorized, because tail handling belongs to `03_denoise`.

The stage reads `<run>/artifacts/01_load_data/panel.parquet` and writes three files to `<run>/artifacts/02_features/`:

| File | Content |
| --- | --- |
| `features.parquet` | `row_id`, `ticker`, `calendardate`, one column per feature, plus auxiliary flags (`neg_equity` and the data-quality flags below) |
| `feature_meta.json` | `features`, `clustering_features`, `auxiliary`, `formulas`, `reasons`, `required_raw_columns`, `data_quality` (rule, rows and tickers caught per flag) |
| `feature_report.json` | non-finite counts, quantiles, pairs with \|Spearman\| >= 0.8, named redundancy checks, and the count of every auxiliary flag |

Run it with `python -m 02_features.utils.cli --run-dir DIR [--config NAME|PATH]` or `scripts/02_features/run.ps1 -RunDir DIR`. The config keys are `features.set` (`baseline12` or `fundamental_v2`) and `features.include_size`, which adds `log_mktcap` to `fundamental_v2`.

## Reporting dimensions in fund.csv

Checked on AAPL 2019-2020 against its filings, and for `netmargin` on the whole panel:

| Dimension | Columns |
| --- | --- |
| Single quarter | `revenue`, `opinc`, `ncfo`, `sps`, `netmargin` |
| Trailing twelve months (TTM) | `netinccmn`, `fcfps` |
| Period-end levels | balance-sheet items |
| Average over the period | `assetsavg` |

`netmargin` is quarterly net income over quarterly revenue. AAPL's 2019Q4 value is 0.242, which is that quarter's $22.2bn net income over $91.8bn revenue; the TTM ratio would be 0.215. On the 25,130-row panel, the sum of four consecutive `netmargin * revenue` values lands within 5% of the TTM `netinccmn` in 98.9% of the 20,523 rows that have four consecutive quarters.

`pb`, `sps` and `fcfps` are stored with 3 decimals.

## `baseline12`: the 12 ratios of the UMAP + DBSCAN baseline

Columns: `pb, netmargin, sps, price, fcfps, quick, roa, roe, ros, pe, debt_eq, equity_rat`. The build divides by `shareswa` first and then takes the ratios, and it keeps the x1000 multipliers on `roa` and `equity_rat`, so the baseline in `baselines/umap_dbscan/` sees exactly the matrix it is defined on.

Regression check: `test_feature_set_matches_stored_checksum` in `src/08_review/tests/test_features.py` rebuilds both feature sets from `fund.csv` on the 25,130-row panel and compares the sha256 of each whole frame (ids, features and flags, in column order) with checksums stored in the test. They were computed on 2026-10-01 and match `features.parquet` of the banked production run (`fundamental_v2`) and of the three baseline runs (`baseline12`). Any change to a formula, a column, the row order or the panel makes the test fail, and every run built on the old features is then out of date. The test skips without the external data or with a different `fund.csv`.

Measured defects (`feature_report.json` of the baseline run `baselines/umap_dbscan/outputs/umap_dbscan--sharadar_sf1_2016q1-2020q3--20261002T070726Z`):

- `ros` is a near-duplicate of `netmargin`: Pearson 0.974, Spearman 0.788, median ratio 3.55.
  - `ros` is TTM net income over quarterly revenue and `netmargin` is quarterly net income over quarterly revenue, so their ratio is TTM over quarterly net income, about 3.5 for a typical company.
  - The high Pearson comes from a few shared extreme outliers; rank agreement is lower because quarterly revenue is noisy.
  - `roa` vs `ros` also have Spearman 0.83.
- `pe` is price over quarterly sales per share, a price-to-sales ratio, although the name suggests P/E.
- The x1000 multipliers on `roa` and `equity_rat` have no effect after standardization.
- Heavy tails dominate a StandardScaler. p01/p99 against min/max:

  | Feature | p01 | p99 | min | max |
  | --- | --- | --- | --- | --- |
  | `pb` | -51 | 69 | -13,478 | 6,930 |
  | `pe` | 0.46 | 1,561 | -24,440 | 1.2 million |
  | `ros` | -146 | 2.2 | -69,700 | 7,099 |

- `price`, `sps` and `fcfps` are per-share levels. They change with share count and splits while the business stays the same.

## `fundamental_v2`: scale-free ratios

Every feature is a ratio of same-currency quantities, so company size and share count cancel. All four valuation features are yields over price; price is always > 0, so yields stay finite where multiples jump between plus and minus infinity. Denominators that must be positive (assets, revenue for margins, current liabilities, equity for ROE) are masked to NaN when they are <= 0, never inverted.

| Feature | Formula | Why |
| --- | --- | --- |
| `book_to_market` | equity / (price * sharesbas) | Inverse of pb from unrounded inputs. Finite and monotone through zero equity. |
| `sales_to_price` | revenue / (price * shareswa), quarterly | Inverse of P/S (`pe` in `baseline12`). Finite at zero sales. |
| `earnings_yield` | netinccmn / (price * shareswa), TTM | Inverse of P/E. No blow-up near zero earnings. |
| `fcf_yield` | fcfps / price, TTM | Valuation on cash rather than accruals. |
| `netmargin` | raw, quarterly net income / quarterly revenue | Profitability per unit of sales. Replaces the redundant `ros`. |
| `operating_margin` | opinc / revenue, quarterly; NaN if revenue <= 0 | Core profitability before financing, tax and one-offs. |
| `roa` | netinccmn / assetsavg; NaN if assetsavg <= 0 | Return on the whole capital base, without the x1000. |
| `roe` *(not clustered)* | netinccmn / equity; NaN if equity <= 0 | Undefined for negative equity, and close to roa / equity_ratio (Spearman roa-roe 0.90). |
| `quick` | (cashneq + investmentsc + receivables) / liabilitiesc | Liquidity without inventory. |
| `current_ratio` | assetsc / liabilitiesc | Liquidity including inventory. |
| `debt_to_assets` | debt / assets | Leverage that stays defined when equity <= 0. |
| `equity_ratio` | equity / assets | Solvency, continuous through zero equity, without the x1000. |
| `asset_turnover` | revenue / assetsavg, quarterly | Capital intensity: separates asset-light from asset-heavy models. |
| `ocf_to_assets` | ncfo / assets, quarterly | Cash profitability, less exposed to accruals than roa. |
| `cash_to_assets` | cashneq / assets | Cash buffer, high for cash-burning growth and biotech firms. |
| `log_mktcap` *(optional)* | log(price * sharesbas) | Size control; off by default so clusters describe business profile. |

Auxiliary column: `neg_equity` (int8) = 1 when equity <= 0. It is never a clustering feature.

## Data-quality flags

Both feature sets carry three more auxiliary `int8` columns from `code/quality.py`. They mark rows whose raw statements look wrong, so a reader or a later filter can find them. They are listed under `auxiliary` in `feature_meta.json` and never under `clustering_features`, so stage 03 does not see them and the clustering is the same with or without them. `feature_meta.json` keeps each rule, the rows and tickers it catches, and any rule skipped because the panel lacks one of its input columns. A row with a missing input, or with a share count or market cap that is not positive, cannot be judged and gets 0.

| Column | Suspect | Rule | Rows / tickers on the 25,130-row panel |
| --- | --- | --- | --- |
| `dq_home_currency_dr` | depositary receipt whose statements are in the home currency while the price is in dollars | the ticker's median of equity / (price * sharesbas) is above 10; every row of the ticker | 57 / 3 (HMC 155.6, TM 52.3, YNDX 14.1; the next ticker, NFH, has 8.3) |
| `dq_share_unit_error` | share count recorded in the wrong unit | a run of at most 4 consecutive rows whose `shareswa` differs by a factor of 10 or more from both the row before and the row after the run, jumping away and back | 4 / 1 (VXRT 2017Q1-Q4: about 136 thousand shares against 3.51 million before and 3.66 million after) |
| `dq_revenue_collapse` | quarterly revenue that collapses while the balance sheet does not | revenue below 1% of the ticker's median revenue (median above 0) while assets stay within 0.5 to 2 times the ticker's median assets | 61 / 48, including DVN 2017Q4 ($3.0 million against a quarterly median of $1.9 billion) |

The share-count rule needs the count to come back. A reverse split or a merger moves the count once and stays, so a single tenfold jump is left alone. AKCA from 2017Q3 (5.7 million weighted shares, then 15.7 and 66.6 million, two steps below tenfold) and DKNG 2019Q4 (50 million, then 706 million, one jump that stays) are not flagged. The revenue rule catches real collapses as well as reporting problems. Of its 61 rows, 43 have negative quarterly revenue, some in the same quarter every year (COTY in the June quarters of 2018, 2019 and 2020). Others are pre-revenue biotechs with sporadic small sales (ACAD, ARWR, RARE) and the cruise lines in 2020Q3 (CCL, NCLH, RCL), whose near-zero revenue is real because they had almost no sailings.

Measured in the banked production run `outputs/bank/clustering/default--sharadar_sf1_2016q1-2020q3--20261002T070710Z/`; the stage log prints the three counts.

## Feature report on the production run

From `feature_report.json` of the same run, on the 25,130-row panel:

- Non-finite values:
  - `roe`: 1,304 rows (the negative-equity rows; `neg_equity` sum = 1,304).
  - `operating_margin`: 51 rows (revenue <= 0).
  - Every other feature: none.
  - 25,079 rows have all 14 clustering features finite.
- Strongly rank-correlated pairs (|Spearman| >= 0.8). These are candidates for down-weighting or dropping after denoising:

  | Pair | Spearman |
  | --- | --- |
  | `roa` and `roe` | 0.90 |
  | `quick` and `current_ratio` | 0.89 |
  | `netmargin` and `operating_margin` | 0.86 |
  | `equity_ratio` and `debt_to_assets` | -0.77 (just below the threshold) |

- Tails remain heavy. `netmargin` and `operating_margin` reach about -20,000 for firms with near-zero revenue. `equity_ratio` reaches 2.66 (equity > assets), which suggests data errors. Robust scaling or winsorization in `03_denoise` is required.
