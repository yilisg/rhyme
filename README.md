# Rhyme

> *History doesn't repeat, but it rhymes.*

Rhyme finds the historical windows that most resemble today's US macro and market environment. It ships with a curated 23-series default panel (FRED + Yahoo), but you can drop in your own CSV or JSON and it will work the same way.

## What it does

1. Pulls a panel of growth, inflation, monetary-financial, and sentiment series.
2. Stationarizes, z-scores (rolling 20y), and aggregates into **four themes**.
3. Slides a **window** across history and builds a feature vector per window
   (moments + cross-series correlations + PCA factor scores).
4. Runs one of two similarity engines:
   - **Primary:** Mahalanobis distance with Ledoit-Wolf shrinkage + Ward hierarchical clustering.
   - **Secondary:** Shape-based distance (k-Shape / SBD) + HDBSCAN.
5. Surfaces:
   - The closest historical analogs to the most-recent window.
   - A **contextual regime label** (Reflation / Stagflation / Goldilocks / Expansion / Slowdown / …).
   - Forward 1m / 3m / 12m returns the analogs produced, for SPX, UST10y, DXY, Baa, Aaa, Gold, and WTI.
   - A UMAP / t-SNE / PCA map of every window.
   - A time-series view with cluster-colored shading going back to 2018.

Full methodology writeup lives in [METHODOLOGY.md](METHODOLOGY.md) and is rendered inside the app.

## Run it locally

```bash
git clone https://github.com/yilisg/rhyme.git
cd rhyme
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The cached default panel (`data/default_panel.parquet`) ships in the repo, so the app works out of the box without a FRED key.

## Refresh the default panel

To pull fresh data from FRED + Yahoo, get a free FRED API key at
<https://fred.stlouisfed.org/docs/api/api_key.html>, copy `.env.example` to
`.env`, paste the key, and run:

```bash
source .venv/bin/activate
python refresh_panel.py
```

The script prints per-series status and rewrites `data/default_panel.parquet`.

## Upload your own panel

In the sidebar, pick **Upload CSV or JSON**. Format:

- First column = date (any parseable format).
- Every other column = numeric time series.

Rhyme resamples to your selected target frequency and applies a heuristic
transform per column. Frequency, rolling-window length, and cluster count
are all adjustable from the sidebar.

## Default series

| Bucket | Series (FRED ID unless noted) |
|---|---|
| **Growth** | initial_claims (ICSA), continuing_claims (CCSA), industrial_production (INDPRO), real_retail_sales (RRSFS), wei (WEI) |
| **Inflation** | core_cpi (CPILFESL), core_pce (PCEPILFE), trimmed_pce_12m (PCETRIM12M159SFRBDAL), be5y5y (T5YIFR), be10y (T10YIE), wti (DCOILWTICO) |
| **Monetary / Financial** | ff (DFF), slope_2s10s (T10Y2Y), slope_3m10y (T10Y3M), nfci (NFCI), vix (VIXCLS), baa_spread (BAA10YM), aaa_spread (AAA10YM), dxy (DTWEXBGS), spx (^GSPC via Yahoo), ust10_yield (DGS10), gold (GC=F via Yahoo) |
| **Sentiment** | umich_sentiment (UMCSENT) |

Note: ICE BofA HY / IG OAS (BAMLH0A0HYM2 / BAMLC0A0CM) are restricted on
the public FRED endpoint to the trailing three years — Rhyme uses the
Moody's Baa/Aaa minus 10y Treasury spreads instead, which have history
back to 1953.

## Deploy to Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in.
2. **Create app** → point at `yilisg/rhyme`, branch `main`, main file `app.py`.
3. Advanced settings → Python **3.11**.
4. (Optional) Add `FRED_API_KEY` as a secret if you want the app to trigger
   a refresh at startup. The shipped parquet works without it.

Every `git push` to `main` redeploys.

## File layout

```
app.py                 # Streamlit UI (6 tabs)
refresh_panel.py       # One-shot script to rebuild the cached panel
rhyme_lib/
  panel.py             # Series specs + parquet loader
  data_fetch.py        # FRED + Yahoo fetchers
  transforms.py        # Stationarize, z-score, theme-aggregate
  features.py          # Window -> feature vector
  similarity.py        # Mahalanobis/Ward + SBD/HDBSCAN, plus embeddings
  labeler.py           # Growth x inflation regime labels
  forward_returns.py   # 1m/3m/12m forward returns off the analog dates
data/
  default_panel.parquet
  default_panel_meta.parquet
sample_data/
  synthetic_macro.csv  # Demo dataset for the "Upload" path
METHODOLOGY.md         # In-app methodology document
requirements.txt
.env.example           # Copy to .env and add FRED_API_KEY
```

## References

Kritzman, Page & Turkington (2012), *Regime Shifts* — the closest public
analog to this tool. Paparrizos & Gravano (2015), *k-Shape*. Campello et
al. (2013), *HDBSCAN*. McInnes et al. (2018), *UMAP*. See
[METHODOLOGY.md](METHODOLOGY.md) for the full bibliography.
