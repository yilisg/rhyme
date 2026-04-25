# Rhyme — engineering notes for Claude

> *History doesn't repeat, but it rhymes.*

A Streamlit dashboard that finds historical analogs for the current macro +
market environment. Live at the user's Streamlit Cloud deployment, source on
GitHub at `yilisg/rhyme`. Deployed branch is **`main`** — pushes to `main`
auto-rebuild within ~30–60s; force a reboot from share.streamlit.io →
Manage app → ⋯ → Reboot if a cache gets stuck.

## What this app actually does

Given a panel of macro + market time series, Rhyme:

1. Resamples to weekly or monthly.
2. Per-series stationarization (level / 3m log-diff / bps-diff / YoY-log /
   pass-through), then rolling 20-year z-score with 10y burn-in.
3. Aggregates z-scores into four themes — **growth, inflation, monetary,
   sentiment** — by equal-weighting within each bucket.
4. Slides an N-period window across the z-scored panel. Each window →
   feature vector: per-series moments (mean, std, skew, ac1, drift) +
   upper-triangle of cross-series correlations + first 3 PCA factor scores.
5. Runs a similarity engine (4 options, see below) to cluster all windows,
   compute distance from today's window to every other window, return the
   top-K analogs.
6. Labels each cluster with a regime tag (Macro mode: growth × inflation
   grid; Market mode: monetary × sentiment grid).

Full prose explanation is in [METHODOLOGY.md](METHODOLOGY.md) — keep that
in sync when the algorithm changes.

## Repo layout

```
rhyme/
├── app.py                  # Streamlit UI, ~1100 lines, 7 tabs
├── refresh_panel.py        # CLI: rebuild data/default_panel.parquet from FRED+Yahoo
├── rhyme_lib/
│   ├── panel.py            # SeriesSpec definitions + parquet loaders
│   ├── data_fetch.py       # FRED + Yahoo fetchers used by refresh_panel
│   ├── transforms.py       # resample, stationarize, z-score, theme aggregate
│   ├── features.py         # window-level feature matrix construction
│   ├── similarity.py       # 4 similarity engines + 2D embeddings
│   ├── labeler.py          # cluster → regime label (mode-specific grids)
│   ├── forward_returns.py  # 1m/3m/12m forward returns for the analogs table
│   └── backtest.py         # walk-forward analog backtest (no look-ahead)
├── data/
│   ├── default_panel.parquet       # cached 32-series panel
│   └── default_panel_meta.parquet
├── .streamlit/config.toml  # base = "light"
├── METHODOLOGY.md          # user-facing methodology + data sources
└── README.md
```

## The default panel (32 series, all free)

Defined in `rhyme_lib/panel.py` as `DEFAULT_SPECS`. To regenerate the cached
parquet (requires `FRED_API_KEY` in `.env` or environment):

```bash
./.venv/bin/python refresh_panel.py
```

Buckets:

- **Growth (10):** ICSA, CCSA, INDPRO, RRSFS, WEI, CFNAI, Empire State,
  Philly Fed, HOUST, USSLIND *(USSLIND ends Feb 2020 — kept for historical
  coverage, auto-excluded from the feature matrix)*
- **Inflation (7):** CPILFESL, PCEPILFE, trimmed PCE, T5YIFR, T10YIE, WTI,
  EXPINF1YR (Cleveland Fed nowcast)
- **Monetary (12):** DFF, T10Y2Y, T10Y3M, NFCI, VIXCLS, BAA10YM, AAA10YM,
  BAMLH0A0HYM2, DTWEXBGS, DGS10, ^GSPC (Yahoo), GC=F (Yahoo) *(HY OAS only
  starts 2023 — auto-excluded from the feature matrix)*
- **Sentiment (3):** UMCSENT, USEPUINDXD, CSCICP03USM665S *(OECD conf is
  ~1 year stale, auto-excluded from the feature matrix)*

**Important: short / stale series filter.** `app.py` (~line 350) drops a
series from the feature matrix if its last obs is > 400 days old or its
first obs is < 10 years before panel end. This is necessary because
`build_window_features` calls `dropna(how="any")` — a single stale series
otherwise collapses the whole NaN-free intersection. Excluded series stay
in theme aggregation so they still contribute to labels.

## Macro vs Market mode

User-toggleable in the sidebar:

| | Macro | Market |
|---|---|---|
| Buckets in feature set | growth + inflation | monetary + sentiment |
| Default freq | M (monthly) | W (weekly) |
| Default window | 60 months | 152 weeks (~3y) |
| Label grid | growth × inflation | monetary × sentiment |
| Label modifier | monetary z (risk-on / risk-off, ±0.5) | VIX z (high vol / calm, ±0.7) |

Themes are still computed over the *full* panel even though only one mode's
buckets feed the feature matrix — this is so the labeler always has all
four themes available.

## Similarity engines (rhyme_lib/similarity.py)

Four genuinely distinct methods, all returning the same `SimilarityResult`
shape:

1. **`primary` (Mahalanobis + Ward)** — Ledoit-Wolf shrunk covariance
   inversion → Mahalanobis distance; Ward agglomerative clustering. The
   default. Closest to Kritzman-Page-Turkington 2012.
2. **`secondary` (SBD + HDBSCAN)** — shape-based distance (1 − max
   normalized cross-correlation) on raw window slices; HDBSCAN density
   clustering, allows "Unclustered" (-1) for novel windows.
3. **`cosine` (Cosine + KMeans)** — direction-only similarity via L2-
   normalized features; spherical KMeans.
4. **`gmm` (GMM log-likelihood signature)** — fits a diagonal GMM, then
   represents each window by its per-component weighted log-likelihood
   `log[π_k · N(x|μ_k, Σ_k)]`, clips at 1%/99% per column, standardizes,
   and uses Euclidean distance between signatures. KEY DETAIL: this is
   genuinely distinct from cosine/Euclidean on raw features — without the
   signature reformulation the GMM and cosine methods produced ~identical
   rankings. See commit df55bd4. Don't accidentally revert to plain
   Euclidean on standardized features here.

## Regime labels (rhyme_lib/labeler.py)

Two grids; pipeline picks one via `mode` parameter.

**Macro grid** (threshold ±0.15, monetary modifier ±0.50):

| | Inflation > +0.15 | neutral | Inflation < −0.15 |
|---|---|---|---|
| **Growth > +0.15** | Reflation | Expansion | Goldilocks |
| neutral | Inflationary | Neutral | Disinflation |
| **Growth < −0.15** | Stagflation | Slowdown | Deflationary bust |

Modifier: monetary z > +0.5 → ` (risk-off)`, < −0.5 → ` (risk-on)`.

**Market grid** (threshold ±0.25, VIX modifier ±0.70):

| | Sentiment > +0.25 | neutral | Sentiment < −0.25 |
|---|---|---|---|
| **Monetary < −0.25** (easy) | Melt-up | Risk-on | Recovery |
| neutral | Bullish | Sideways | Cautious |
| **Monetary > +0.25** (tight) | Tightening peak | Risk-off | Crisis |

Modifier: VIX z > +0.7 → ` (high vol)`, < −0.7 → ` (calm)`.

`label_from_z(...)` is a public entry point exposed for sibling projects
(accent, etc.) that want a point-in-time label without running the whole
pipeline. Don't break that signature.

**Known data limitation:** Macro mode currently shows mostly
"Inflationary" / "Stagflation" labels because the feature-set NaN-free
intersection only spans ~42 monthly windows ending 2022–2025, all
genuinely inflation-dominated. Tightening the threshold from 0.30 → 0.15
helped marginally. The right long-term fix is more pre-2010 inflation
coverage with series that don't have NaN gaps; the short-term fix is the
3-clock Cycle tab which lets the user see growth/inflation, vol/valuation,
and sentiment/stress directly without depending on cluster labels.

## The Cycle tab (3 clocks)

`app.py` ~line 493+. Shared `_draw_clock` helper renders a 4-quadrant
scatter with quadrant tints, history dots, landmark events (diamond
markers from `CYCLE_EVENTS`), and a gold T-12M → Today trail.

1. **Macro** — themes['growth'] × themes['inflation']. Quadrants:
   Reflation / Stagflation / Goldilocks / Deflationary bust.
2. **Market** — z['vix_z'] × mean(z['baa_spread_z'], z['aaa_spread_z']).
   Quadrants: Panic / Calm but cheap / Topping / Melt-up.
3. **Sentiment** — z['umich_sentiment_z'] × mean(z[stress_pool]) where
   `stress_pool = [nfci_z, vix_z, baa_spread_z]` filtered to what's
   present. Quadrants: Disbelief / Fear / Euphoria / Apathy.

Landmark text color is `rgba(60,60,60,0.9)` — dark enough for the light
theme. If you ever flip back to dark mode, that needs to change.

## UI conventions

- House style locked in `~/.claude/projects/.../memory/project_rhyme_ui_pattern.md`
  (sidebar + tabs + Methodology tab + gold/crimson accents).
- Distance values displayed to 3 decimals (`f"{v:.3f}"`) — e.g. 0.450, not
  0.45. Don't drop trailing zeros.
- Theme: light. Configured in `.streamlit/config.toml`. Sibling tools
  `accent` and `compose` are also light.
- Tabs: Overview · Data · Cycle · Regime map · Time series · Analogs ·
  Methodology.

## Caching

`@st.cache_data` decorates `_cached_default_panel`, `_cached_pipeline`,
and `_cached_walk_forward`. Cache keys are tuples of plain strings/ints
(method, mode, window, robust, etc.) plus a hash-of-the-panel-bytes.
Streamlit Cloud reboots clear `st.cache_data`. To force a clean run
locally, delete `~/.streamlit/cache` or rerun `streamlit run app.py`
after a code change.

## Testing / smoke check

There's no formal test suite. The de-facto smoke test is:

```bash
cd /Users/yili/Desktop/Claude/rhyme
./.venv/bin/streamlit run app.py --server.headless true --server.port 8511
# verify HTTP 200 at http://localhost:8511 and no traceback in the console
```

Quick end-to-end sanity check from Python (pipeline only, no UI):

```python
from rhyme_lib.panel import load_default_panel, DEFAULT_TRANSFORMS, DEFAULT_BUCKETS, DEFAULT_SPECS
from rhyme_lib.transforms import transform_and_zscore, theme_aggregate
from rhyme_lib.features import build_window_features
from rhyme_lib.similarity import primary_similarity
from rhyme_lib.labeler import label_clusters

panel, meta = load_default_panel()
codes = [s.code for s in DEFAULT_SPECS if s.bucket in {"growth", "inflation"}]
z = transform_and_zscore(panel, DEFAULT_TRANSFORMS, freq="M",
                         rolling_years=20, min_years=10)
themes = theme_aggregate(z, DEFAULT_BUCKETS)
# Drop stale/short series (replicates the app.py filter)
panel_end = panel.index.max()
keep = [c for c in codes
        if not panel[c].dropna().empty
        and panel[c].dropna().index.max() >= panel_end - pd.Timedelta(days=400)
        and panel[c].dropna().index.min() <= panel_end - pd.Timedelta(days=365*10)]
zcols = [f"{c}_z" for c in keep if f"{c}_z" in z.columns]
wf = build_window_features(z[zcols], window_size=60)
res = primary_similarity(wf, n_clusters=5)
labels = label_clusters(themes, res.labels, wf.end_dates, mode="macro", individual_z=z)
```

If this raises `not enough NaN-free rows` something has changed in the
panel — typically a new short-history series was added without a
corresponding entry in the auto-exclude filter.

## Recent change history (most recent first)

- **89cbb55** Default to light theme (.streamlit/config.toml + darken
  Cycle landmark labels for readability).
- **8b79fb3** Mode-specific regime labels (Market grid added), 3-clock
  Cycle tab, panel expanded 23 → 32 free series, METHODOLOGY data-sources
  section, auto-exclude stale/short series from feature matrix.
- **df55bd4** GMM similarity reformulated to log-likelihood signatures
  (was ~identical to cosine on standardized features); distance display
  format pinned to 3 decimals.
- **3647423** Walk-forward backtest with no look-ahead.
- **b3a1b3a** Cycle tab v1, robust mode (median/MAD/Spearman), two new
  similarity methods (cosine + gmm), analog backtest.
- **54f94d9** v3: Macro/Market mode toggle, dark-mode CSS fixes (later
  superseded), analog table rework.

PRs: #1 merged (v2 panel/methodology), #2 merged (v3 ui-polish-and-mode-
toggle, the bulk of the recent work).

## Conventions for future work

- **Don't add tests/* or docs/* on speculation** — keep changes tight and
  in-place. Project owner prefers minimal one-word names with short
  italic taglines (see user_finance_profile memory).
- **Don't break `label_from_z` signature** — sibling project `accent`
  imports it.
- **Always 3-decimal distances** in any UI surface.
- **Themes computed over full panel; features computed only over the
  active mode's buckets.** Don't conflate the two scopes.
- **`dropna(how="any")` in `build_window_features` is a hard constraint** —
  always filter `feature_codes` for staleness/length before passing in.
- **Light theme everywhere** — match accent and compose. If you ever
  introduce dark-only colors (e.g. `rgba(220,220,220,...)` for text),
  audit that they read on white.
- **Streamlit Cloud auto-deploys from `main`.** Push to `main` =
  production deploy. Use feature branches + PRs for anything non-trivial.

## Outstanding ideas / not yet wired up

Documented in METHODOLOGY.md "Suggested but not yet included" section:
ISM PMI sub-components (paid), Atlanta Fed GDPNow (free CSV), NY Fed
Nowcast (free), Caldara-Iacoviello GPR (free CSV), MOVE index, Conference
Board LEI (paid), CFTC COT (free, needs cleaning), Citi Surprise (paid).
GDPNow and GPR are the two highest-value free additions if someone
wants to plumb them through `data_fetch.py`.
