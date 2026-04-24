# Rhyme

> *History doesn't repeat, but it rhymes.*

Upload a panel of macro and market time series. Rhyme slides a rolling window across history, clusters the windows by their statistical similarity, and tells you which historical environments today's window most resembles.

## Input format

A CSV or JSON file where:

- The **first column is a date** (any parseable format).
- Every other column is a **numeric series** — daily or weekly returns, yield changes, spread levels, vol, etc.

Example:

```
date,spx_ret,ust10_chg,hy_oas,vix
2020-01-02,0.0084,-0.02,3.36,12.47
2020-01-03,-0.0076,-0.05,3.38,14.02
...
```

A synthetic example lives in [`sample_data/synthetic_macro.csv`](sample_data/synthetic_macro.csv).

## Run it locally

```bash
git clone https://github.com/yilisg/rhyme.git
cd rhyme
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`. Upload your CSV via the sidebar.

## What it computes

1. **Windowing** — slides a window of `N` rows across the panel. Each window is standardized per-series, then flattened into a feature vector.
2. **Clustering** — KMeans on the window feature matrix, giving each window a regime label.
3. **Analogs** — Euclidean distance from the reference (most-recent) window's feature vector to every historical window; the closest `k` are reported.
4. **Visuals** — a 2D PCA projection of the window space (colored by cluster, reference marked with a star), plus a per-series overlay of the reference window against its top analogs.

## Parameters

| Sidebar control | Meaning |
|---|---|
| Window size | Length of each rolling window, in rows of the input |
| Number of regime clusters | `k` for KMeans |
| Top analogs to show | How many nearest historical windows to list |

## File layout

```
app.py            # Streamlit UI
analysis.py       # Windowing, clustering, analog search
requirements.txt
sample_data/
  synthetic_macro.csv
```
