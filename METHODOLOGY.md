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

Rhyme has two mode-specific labelers. The pipeline picks whichever matches
the selected mode.

### Macro mode — growth × inflation grid (Merrill Lynch clock)

For each cluster, Rhyme computes the mean z-score across member windows on
Growth, Inflation, and Monetary/Financial themes. The base label comes
from the 2D grid of (growth sign, inflation sign):

|                     | Inflation > +0.15 | Neutral       | Inflation < −0.15 |
|---------------------|-------------------|---------------|-------------------|
| **Growth > +0.15**  | Reflation         | Expansion     | Goldilocks        |
| **Neutral**         | Inflationary      | Neutral       | Disinflation      |
| **Growth < −0.15**  | Stagflation       | Slowdown      | Deflationary bust |

A modifier is appended when the Monetary/Financial z-score exceeds |0.5|:

- `(risk-off)` when the financial theme is stressed (wide spreads, high vol, tight policy)
- `(risk-on)` when the financial theme is easy

### Market mode — monetary × sentiment grid

Market mode clusters on monetary + sentiment series, so the label grid
uses those axes instead:

|                     | Sentiment > +0.25 | Neutral       | Sentiment < −0.25 |
|---------------------|-------------------|---------------|-------------------|
| **Monetary < −0.25** (easy) | Melt-up   | Risk-on       | Recovery          |
| **Neutral**         | Bullish           | Sideways      | Cautious          |
| **Monetary > +0.25** (tight/stressed) | Tightening peak | Risk-off | Crisis |

A modifier is appended when the VIX z-score exceeds |0.7|:

- `(high vol)` when VIX is elevated
- `(calm)` when VIX is unusually low (often complacency signal)

These are mnemonics, not predictions — they are derived from cluster means alone.

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

### NaN-tolerant feature matrix

`build_window_features` does **not** require an NaN-free intersection
across series. For each window, every column with at least
`max(window // 2, 6)` non-NaN observations contributes its moments;
columns inactive in that window simply yield NaN moment entries.
Cross-series correlation pairs use the per-pair valid intersection.
The similarity engines (Mahalanobis, cosine, GMM) standardize NaN-aware
on the column-mean and impute at zero (≡ column mean in z-space) for
clustering, but distance computations restrict to the per-row active-
mask intersection so a window with limited coverage isn't artificially
close to anything. Windows with fewer than 5 active columns are
dropped. This replaces the old `dropna(how="any")` collapse and
expanded the default monthly Macro window count from ~42 to ~562.

### Long-term-model toggle

The sidebar **Long-term model** checkbox (default off) lets the user
include panel observations before 2000-01-01. With the checkbox off,
the panel is filtered to `>= 2000-01-01` regardless of source — this
matches the practical coverage floor of the public FRED+Yahoo panel
where most series start ~2000. With the checkbox on, deep-history
sources (e.g. the tabula GFD-sourced parquet) flow through unmodified.

## Optional advanced analyses

### Bayesian regime probabilities

The Overview tab renders a softmax over Mahalanobis distance to each
cluster centroid: `p_k ∝ exp(-d_k² / τ)`. The temperature τ defaults to
the median within-cluster squared Mahalanobis distance, so a window
exactly at the typical within-cluster spread of cluster k is at probability
mass `e^{-1}` relative to cluster k's central tendency. This replaces
the binary "today is regime X" badge with a richer view of how
ambiguous today's classification is.

### Walk-forward regime labels (sidebar toggle)

By default clusters are fit on the full panel up to today, anchoring
the label space to one snapshot. With the toggle on, clusters are
refit every 12 periods on a 360-period (or full-available) trailing
window; each window's label is whichever centroid (under the
contemporaneous fit) it's closest to. This costs an order of magnitude
more compute but reflects how the regime would have been labeled at
the time, not in 2026 hindsight.

### Hierarchical similarity (3y / 10y / 30y)

The same Mahalanobis distance, but recomputed at three different
window lengths. Today's regime can rhyme with different histories on
different timescales — a 3-year cyclical match might point at one
era while a 30-year structural match points at another. Renders as a
per-horizon top-5 table in the Analogs tab.

### Block-bootstrap significance

Permutes the panel in 24-period blocks 1000× and records the best-
analog Mahalanobis distance under each permutation. The result is a
null distribution for "best-analog distance under random data". If
today's actual best-analog distance is in the small-tail of this null
(say, <5th percentile), the analog is meaningfully closer than
chance; if it's near the median, today doesn't rhyme with anything
in particular more than random data would.

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

## Data sources

### Currently in the default panel (all free)

Growth (from FRED):
- `ICSA`, `CCSA` — Initial and continuing jobless claims (weekly)
- `INDPRO` — Industrial production (monthly)
- `RRSFS` — Real retail & food services sales (monthly)
- `WEI` — NY Fed Weekly Economic Index (weekly, from 2008)
- `CFNAI` — Chicago Fed National Activity Index (monthly, from 1967)
- `GACDISA066MSFRBNY` — NY Fed Empire State manufacturing general business (monthly, from 2001)
- `GACDFSA066MSFRBPHI` — Philly Fed manufacturing general activity (monthly, from 1968)
- `HOUST` — Housing starts (monthly)
- `USSLIND` — US Leading Index (monthly, *discontinued Feb 2020* — retained for historical coverage)

Inflation (from FRED):
- `CPILFESL`, `PCEPILFE` — Core CPI and Core PCE (monthly)
- `PCETRIM12M159SFRBDAL` — Dallas Fed trimmed-mean PCE, 12m (monthly)
- `T5YIFR`, `T10YIE` — 5y5y and 10y breakeven inflation (daily)
- `DCOILWTICO` — WTI crude oil spot (daily)
- `EXPINF1YR` — Cleveland Fed 1-year expected inflation nowcast (monthly, from 1982)

Monetary / financial (from FRED + Yahoo):
- `DFF` — Effective fed funds rate (daily)
- `T10Y2Y`, `T10Y3M` — Yield curve slopes (daily)
- `NFCI` — Chicago Fed financial conditions (weekly)
- `VIXCLS` — VIX implied volatility (daily)
- `BAA10YM`, `AAA10YM` — Moody's credit spreads vs. 10y (monthly)
- `BAMLH0A0HYM2` — ICE BofA US High Yield OAS (daily)
- `DTWEXBGS` — Trade-weighted USD broad (daily)
- `DGS10` — 10y Treasury yield (daily)
- `^GSPC` (Yahoo) — S&P 500
- `GC=F` (Yahoo) — Gold front-month futures

Sentiment (from FRED):
- `UMCSENT` — University of Michigan consumer sentiment (monthly)
- `USEPUINDXD` — Baker-Bloom-Davis US Economic Policy Uncertainty (daily, from 1985)
- `CSCICP03USM665S` — OECD Consumer Confidence Indicator, US (monthly)

### Suggested but not yet included

**ISM Manufacturing / Services PMI** (licensed, paid via ISM or Markit) —
the canonical US activity surveys. Sub-components like ISM Prices Paid
are excellent inflation leading indicators. ISM used to publish the
headline index on FRED but the licensing arrangement ended around 2018,
so the series no longer updates there. Alternatives in-panel today:
Chicago Fed NAI (CFNAI), NY Fed Empire State, Philly Fed manufacturing —
together they approximate the ISM composite reasonably well. To ingest
ISM directly you would need a paid ISM report subscription or a Markit
PMI feed (source: S&P Global Market Intelligence).

**Atlanta Fed GDPNow** — real-time nowcast of current-quarter real GDP
growth. Published at https://www.atlantafed.org/cqer/research/gdpnow as
CSV. Free but not on FRED, so not yet wired into the default panel. Add
as a custom series via upload, or plumb through data_fetch.py.

**NY Fed Nowcast** — similar to GDPNow but from the NY Fed:
https://www.newyorkfed.org/research/policy/nowcast

**Geopolitical Risk (GPR) index** (Caldara & Iacoviello) — monthly and
daily series at https://www.matteoiacoviello.com/gpr.htm. Free, CSV
download, updates monthly. A natural complement to EPU for the
sentiment clock.

**MOVE index** — Treasury implied volatility analogue to VIX. Available
via ICE / Bloomberg as `^MOVE` on Yahoo Finance, though history is
inconsistent. Valuable for a rates-stress signal.

**Global PMI composites** — IMF and S&P Global publish these, generally
behind paywalls. Free proxy: aggregate ECB, BoJ, and PBoC official PMIs.

**Conference Board Leading Economic Index (LEI)** — proprietary, monthly
release. Historical series available via Conference Board subscription
or Haver. Similar information to CFNAI + USSLIND — partial substitute.

**Commodity-specific vol & positioning** — COT (Commitment of Traders)
data from CFTC, weekly, free. Useful for positioning-based sentiment but
requires some cleaning.

**Corporate earnings revisions & surprise indices** — Citi Economic
Surprise Indices are the standard, Bloomberg/Citi proprietary. Free
alternative: Atlanta Fed's inflation expectation surveys, or FRED's
`MICH` (UMich 1y inflation expectations).

## References

- Kritzman, Page & Turkington (2012). *Regime Shifts: Implications for Dynamic Strategies.* Financial Analysts Journal.
- Paparrizos & Gravano (2015). *k-Shape: Efficient and Accurate Clustering of Time Series.* SIGMOD.
- Campello, Moulavi, Sander (2013). *Density-based clustering based on hierarchical density estimates.* PAKDD.
- Aghabozorgi, Shirkhorshidi & Wah (2015). *Time-series clustering — a decade review.*
- Hamilton (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica.
- Ang & Timmermann (2012). *Regime Changes and Financial Markets.*
- McInnes, Healy & Melville (2018). *UMAP: Uniform Manifold Approximation and Projection.*
