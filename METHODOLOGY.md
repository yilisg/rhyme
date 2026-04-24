## What Rhyme does

Given a panel of macro and market time series, Rhyme:

1. Resamples every series to a common frequency (weekly default, monthly option).
2. Applies a per-series **stationarization**: levels for already-stationary
   spreads, 3-month log-difference for price/index series, basis-point difference
   for rates and yields, year-over-year log-difference for flow series, and pass-through
   for series that are already expressed as year-over-year rates.
3. Z-scores each transformed series against a rolling 20-year window (with
   a 10-year minimum history before the first z-score is produced).
4. Aggregates the z-scored series into four **themes** — Growth, Inflation,
   Monetary/Financial, Sentiment — by equal-weighting within each bucket.
5. Slides a **window** of N periods across the z-scored panel. For each window,
   extracts a feature vector:
   - per-series moments: mean, standard deviation, skewness, lag-1 autocorrelation, start-to-end drift
   - upper triangle of the within-window cross-series correlation matrix
   - first three PCA factor scores of the window's raw z-values
6. Runs one of two similarity engines (see below) and returns (a) a cluster
   label for every historical window, (b) a distance from the most-recent
   window to every other window, and (c) the top-K closest historical windows.

## Similarity engines

### Primary: Mahalanobis distance + Ward hierarchical clustering

- Standardize the feature matrix; compute a **Ledoit-Wolf shrunk covariance**
  of the features; invert to get a Mahalanobis metric. Distance from today
  to each historical window is the Mahalanobis distance in that space.
- Cluster with **Ward linkage** (minimum-variance agglomerative). Deterministic,
  dendrogram-interpretable, no random seed.

**Why:** feature moments are strongly correlated (mean and drift, std and
skew, correlation entries). Plain Euclidean double-counts those dimensions;
Mahalanobis de-correlates them. Ward clusters are defensible — "today is
grouped with Q4 2015 because they share a sub-tree at height X" is a
narrative a strategist can work with.

This approach is closest in spirit to Kritzman, Page & Turkington,
*"Regime Shifts: Implications for Dynamic Strategies"*, Financial Analysts
Journal, May/June 2012 — which uses Mahalanobis turbulence on asset
returns. Rhyme extends it to a themed macro + market panel and to window
similarity rather than point-in-time turbulence.

### Secondary: Shape-based distance + HDBSCAN

- **SBD** (shape-based distance; Paparrizos & Gravano, *k-Shape*, SIGMOD 2015)
  = 1 − max normalized cross-correlation, averaged across series within a window.
  Nearly as cheap as Euclidean, but invariant to small phase shifts in the
  underlying paths.
- **HDBSCAN** (Campello, Moulavi, Sander 2013): density-based, does not
  require you to pick k, and flags windows that do not belong to any cluster
  as noise (label = −1 → "Unclustered"). Valuable honesty when the
  environment is genuinely novel.

Use this when you want to emphasize the *shape* of how the environment
developed within each window rather than summary statistics, or when you
want the model to tell you that today doesn't rhyme with anything.

### Why not DTW?

Dynamic Time Warping lets you align two time series by stretching and
compressing the time axis. It is the right tool when the signals you're
matching are genuinely asynchronous (ECG beats, speech). For multivariate
macro and market panels, all series share the same calendar — Feb 2008
and Sep 2008 are different windows by design, not two versions of the
same window seen at different speeds. Applying DTW per-series would also
destroy lead-lag structure between series, and multivariate DTW is
O(N² · d). Skipped deliberately.

## Regime labels

For each cluster, Rhyme computes the mean z-score across member windows on
Growth, Inflation, and Monetary/Financial themes. A label comes from the
2D grid of (growth sign, inflation sign):

|                    | Inflation > +0.3 | Neutral       | Inflation < −0.3 |
|--------------------|------------------|---------------|------------------|
| **Growth > +0.3**  | Reflation        | Expansion     | Goldilocks       |
| **Neutral**        | Inflationary     | Neutral       | Disinflation     |
| **Growth < −0.3**  | Stagflation      | Slowdown      | Deflationary bust |

A modifier is appended when the Monetary/Financial z-score exceeds |0.5|:

- `(risk-off)` when the financial theme is in stressed territory (wide spreads, high vol, or tight policy)
- `(risk-on)` when the financial theme is in easy territory

These are mnemonics, not predictions — they are derived from the cluster
means alone.

## Data harmonization

The default panel is stored at native frequency (macro monthly, markets
weekly/daily) and resampled to the target frequency at feature-build time.
Forward-filling is bounded (at most 30 days on the underlying daily panel,
and at most 6 periods after resampling) to avoid creating fake
autocorrelation that would corrupt the moment features.

The macro z-score window is **rolling 20 years** by default — long enough
to span a full cycle, short enough to adapt to regime shifts like the
post-GFC low-rate environment. An **expanding** option is available for
maximally long history at the cost of over-weighting the 1970s.

## Visualization choices

- **Scatter (regime map):** UMAP is the default because it preserves global
  structure better than t-SNE while remaining as fast. t-SNE and PCA are
  exposed as toggles; t-SNE carries an explicit caption warning that
  cluster-to-cluster distances in t-SNE plots are not meaningful.
- **Time-series view:** theme z-score lines overlaid with cluster-colored
  vertical bands to show regime persistence, plus gray NBER recession bars
  for context.
- **Analogs table:** top-K windows ranked by distance, joined with forward
  1m / 3m / 12m returns for SPX, UST 10y yield change, DXY, Baa and Aaa
  credit spreads, gold, and WTI.

## References

- Kritzman, Page & Turkington (2012). *Regime Shifts: Implications for Dynamic Strategies.* Financial Analysts Journal.
- Paparrizos & Gravano (2015). *k-Shape: Efficient and Accurate Clustering of Time Series.* SIGMOD.
- Campello, Moulavi, Sander (2013). *Density-based clustering based on hierarchical density estimates.* PAKDD.
- Aghabozorgi, Shirkhorshidi & Wah (2015). *Time-series clustering — a decade review.*
- Hamilton (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica.
- Ang & Timmermann (2012). *Regime Changes and Financial Markets.*
- McInnes, Healy & Melville (2018). *UMAP: Uniform Manifold Approximation and Projection.*
