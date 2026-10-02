# 01_load_data

Reads the SHARADAR SF1 quarterly extract and applies the missing-data policy. It also writes a report on what was dropped and why.

- Inputs (external data root, `FC_DATA_DIR`): `raw/fund.csv` and `raw/SHARADAR_INDICATORS_a3407c5c2ec46991d2b7ac667785d0d1.csv`.
- Outputs (`<run>/artifacts/01_load_data/`):
  - `panel.parquet`: `row_id` (0-based row position in fund.csv; the baseline carries the same id), `ticker`, `calendardate`/`datekey` (datetime64), and all 20 raw numeric columns.
  - `missing_report.json`: what the filters dropped and why.
  - `indicator_definitions.parquet`: title, description and unit for each column.
- Entry points:
  - `python -m 01_load_data.utils.cli [--run-dir DIR] [--config NAME|PATH]`
  - `scripts/01_load_data/run.ps1 [-RunDir DIR] [-Config NAME|PATH]`
- Config:
  - `data.drop_calendardates`: default `["2020-12-31"]`. Only 44 companies had reported 2020Q4.
  - `missing.policy`: one of
    - `all_columns`: drops a row with a NaN in any raw column; the default, also used by the baseline;
    - `feature_columns`: drops only rows missing a raw input of the selected feature set;
    - `none`.
  - `missing.financial_probe_tickers`.

## Measured on fund.csv (sha256 4782078f...cd9)

Run: `outputs/bank/clustering/default--sharadar_sf1_2016q1-2020q3--20261002T070710Z` (`artifacts/01_load_data/missing_report.json`).

- Rows: 35,053 in the file. Dropping 2020-12-31 removes 44, leaving 35,009. The policy then drops 9,879, leaving 25,130 rows and 1,474 tickers. The median kept ticker has 19 quarters.
- Tickers: 454 of 1,928 are dropped entirely and 303 are dropped partially.
- NaN patterns: the most common pattern among dropped rows is `assetsc+liabilitiesc+investmentsc` (6,888 rows). These are the classified balance-sheet columns, which banks, insurers and many REITs do not report.
- The financial-firm hypothesis is supported.
  - 376 tickers have `assetsc` and `liabilitiesc` missing in every quarter. All of them are dropped entirely, and they make up 82.8% of the dropped tickers.
  - These tickers have median assets of $10.5bn and median equity/assets of 0.22, against $3.1bn and 0.41 for kept tickers, the profile of balance-sheet-heavy financial firms.
  - Of the 17 probe tickers, 16 are in the file and 15 are dropped entirely: JPM, BAC, WFC, C, GS, MS, USB, PNC, AIG, MET, PRU, ALL, TRV, SPG and PLD.
  - AMT, a tower REIT that does report a classified balance sheet, is kept. BRK.B is not in the file.
- Every clustering built on this panel describes non-financial firms only, a selection bias that leaves out banks and insurers.
- The `feature_columns` policy changes little: it keeps 25,131 rows for `fundamental_v2` and 25,447 for `baseline12` (`load_panel` with that policy on the same file). Both sets use `quick`, which needs the classified balance-sheet columns. Including financial firms would take a feature set without `quick`/`current_ratio`, and their revenue and leverage concepts are not comparable to industrial firms.
