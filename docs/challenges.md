# Challenges and how I solved them

The project groups 1,474 companies by accounting ratios from 25,130 quarterly reports (2016Q1 to 2020Q3, SHARADAR SF1). The first pipeline was UMAP + DBSCAN: 12 ratios through StandardScaler, a 2-D UMAP with the correlation metric, and DBSCAN on that plane. It stays in the repository as the comparison baseline. The final pipeline sorts companies into five archetypes with k-means on profiles of 14 scale-free ratios. The runs behind every number are cited in the READMEs under `src/`, `experiment/` and `baselines/umap_dbscan/`.

## A few rows owned the variance

After StandardScaler, the top 1% of rows held 80% of the sum of squares of the 12 baseline ratios, against 3% for a Gaussian with the same covariance. My first fix, winsorizing at 1% and 99% and then scaling by median and IQR, made things worse, because an IQR says nothing about how far a tail reaches. On the 14 production ratios, net margin's clipped 1st percentile still sat 349 IQRs out, the two margin columns carried 99.1% of the variance, and k-means put 93% to 97% of the rows in one cluster at a stability ARI of 0.79 to 0.99. Taking asinh of the IQR-scaled value, linear for ordinary firms and logarithmic in the tails, cut the top-1% share to about 10%. Against the StandardScaler space of the same 14 ratios (four clusterers, five seeds), stability rose by 0.13 and the separation of return volatility across clusters rose from epsilon squared 0.119 to 0.185 with one unit per company and cluster (0.164 to 0.282 in the verification, with one unit per company), up in every matched pair. Outlier-only clusters disappeared. No other denoising step moved stability by more than 0.021 on these ratios, or 0.043 on the baseline's 12. Stage 03 now writes the top rows' share of the sum of squares into every run's `denoise_meta.json`, and the selection rule reads stability only together with a cap on the largest cluster.

## Repeated quarters of the same company

The median company contributes 19 similar quarters. A rank intraclass correlation of 0.76 means a design effect near 13, so the 25,130 rows carry about as much information as 1,900 independent ones. Under random labels assigned per company, row-level Welch t-tests rejected 49% of the time and a row-level Kruskal-Wallis test on the size of next-quarter returns 95.5%. A classifier scored 0.62 on those random labels with a row split and 0.06 with whole companies held out.

Every test now uses one unit per company: company medians, compared with Mann-Whitney tests under Benjamini-Hochberg correction and sized with Cliff's delta. Under the same null these reject 4.5% and 2.5%. Every resampling and permutation step moves whole companies, and the final model is fitted on one median profile per company. Keeping one row per company moved the average silhouette by about 0.01 (at most 0.014), so internal scores give no warning, and company grouping has to be built into the tests from the start.

## Good scores that measured something else

The baseline's tuning grid reports silhouettes of 0.404, 0.342 and 0.327 for DBSCAN on the 12 standardized ratios, which splits off a group of 478 to 839 rows, all with zero or negative equity, from 66 to 110 companies, and scores noise as a cluster. The 17 clusters the baseline delivers score 0.279 in their UMAP plane, 0.154 with noise counted and -0.078 in the 12-D space. Persistence can be gamed outright: labels constant within each company reach a kappa of 1.000, above every real clustering (0.65 to 0.88).

I now score in the space the clusterer saw, report noise separately and quote outside scores as excess over a company-level permutation null. Selection has two layers. A model must first reproduce on unseen companies (stability ARI 0.6, size-weighted prediction strength 0.8, seed ARI 0.9, at most 20% noise, no cluster above 60%), and survivors are ranked by how well they separate next-quarter return size and volatility beyond a within-sector null. Over 31 candidates this chose k-means with K = 5 in every seed and resample, and the baseline failed three thresholds in every seed. Silhouette alone would have picked K = 3 with 73% of the rows in one cluster.

My first rule averaged two ranks, and a 0.002 NMI edge flipped one resample to K = 4, so NMI now decides only within 0.01, the resampling spread of one model's economic score.

## No natural number of clusters

Under k-means the company profiles compress better than a Gaussian copula null at every K from 2 to 10, yet nothing in the data named a K. The gap statistic stopped at K = 2, a 90/10 split isolating the loss-makers, and at K = 1 once the 334 loss-making companies were removed.

I treated the clusters as a quantization of a skewed continuum and chose K by reproducibility. For k-means on 2016 to 2019 profiles, size-weighted prediction strength was 0.89, 0.87 and 0.77 at K = 4, 5 and 6, and 0.49 to 0.63 above that, while K = 4 put 55% of the rows in one cluster. Economic separation stayed flat above K = 5 (0.223 at K = 5, 0.204 to 0.214 up to K = 12), so finer partitions describe returns about as well, yet they do not come back when refit on other companies.

K = 6 passed at one seed of three (0.804), so I take the worst case over seeds and report that any threshold from 0.75 to 0.85 keeps K = 5.

## UMAP islands that persistence alone produces

The baseline's 17 to 20 islands looked like business models, but on average a fifth of a row's 10 nearest neighbours in its plane are the same company's other quarters (chance 0.07%). Column-permuted and Gaussian nulls gave 1 to 5 clusters, too easy a test for a method built on a neighbour graph, so I used a ticker-block null that hands each company's series of every column to another company: it keeps persistence and breaks the pairing of ratios. It gave 11 to 17 islands with a similar noise share. The real panel still separated better, with a plane silhouette of 0.21 to 0.28 against -0.19 to 0.11.

For interpretation I used correlation PCA on the tail-compressed ratios, where parallel analysis kept four components in every check; they explain 74.5% of held-out companies' variance. The axes read as profitability, balance-sheet strength, value and asset turnover. I still cluster on the 14 ratios: the k-means models that reproduce on the PC scores, K = 6 and in the verification K = 4, separate next-quarter moves less well (economic excess 0.190 and 0.162 to 0.166 against 0.222).

## Data problems the filters hid

Dropping rows with any missing value removed 454 of 1,928 tickers, mostly firms without a classified balance sheet; 15 of the 16 banks, insurers and REITs I probed were gone. HMC, TM and YNDX report in their home currency against a dollar price, so their median book-to-market runs from 14 to 156, and VXRT's 2017 share count reads about 136 thousand against 3.51 million around it.

After tail compression the outlier vote flagged none of the 38 TM and HMC quarters. So I wrote rules in the feature stage: a home-currency depositary receipt flag, a share-unit flag that needs the count to jump tenfold and come back (sparing reverse splits), and a revenue-collapse flag, all auxiliary columns outside the clustering. Deleting would lose real events, since the revenue rule also catches the cruise lines in 2020Q3, whose sales did vanish. For financial firms I kept the filter and limited every claim to non-financial companies.

The detectors only measure distance, and a company whose quarters are all wrong in the same way forms a dense group that none of them flags. I would write rules like these before running any detector.

## Seeded UMAP that changed with the thread count

Jobs ran with 8 or 2 threads on a shared machine. With one seed, UMAP gave a different clustering with OpenBLAS at 8 threads than at 2. Each UMAP call now runs under a one-thread limit on every native pool through threadpoolctl. Embeddings are bit-identical with 8 and 2 threads, and every refit at a run's own seed reproduced the delivered clustering exactly. A regression test checks bitwise equality under 1 and 4 threads, because a nearly equal embedding still moves DBSCAN's clusters.

Isomap also differed between two runs and has no limit yet. I would test every stochastic step across thread counts before the first study run.

## Keeping the return check honest

The archetypes looked like a risk signal, and two things could fake one. Prices are fiscal period-end closes, so a window from quarter t to t+1 starts before the report is filed. The archetypes also differ in size (in the banked run, a median market cap of $6.9 billion for mature leveraged firms and $1.4 billion for pre-revenue ones), and small stocks move more.

Each absolute price change is measured against the quarter's cross-sectional median, so a market-wide crash adds nothing. Epsilon squared across archetypes, one value per company, was 0.286 against a within-sector null of 0.024, and 0.272 on the post-filing window from t+1 to t+2. With sector, quarter and size-decile fixed effects and ticker-clustered errors, the archetype added a partial R squared of 2.8%. That became 1.5% to 1.8% with trailing volatility as a control, and 2.93% when archetypes fitted on 2016 to 2017 were tested from 2018. The two previous price moves explain slightly more (3.0% to 3.6%), and the archetype adds 1.7% to 1.9% beyond them, so I report an association with risk and make no trading claim.

Prices carry no dividends, and delisted companies, more often loss-making, drop out of the forward returns, so the gaps are probably understated; total returns that include delisting would settle it.
