"""Rhyme — historical analog / regime detection for US macro + market panels.

Six tabs:
  1. Overview    — headline: today's regime, top analogs, what they imply
  2. Data        — panel preview, upload/replace, bucket filter, date range
  3. Regime map  — UMAP / t-SNE / PCA scatter, reference starred
  4. Time series — theme z-scores + cluster-shaded timeline + recession bars
  5. Analogs     — top-K table with forward 1m/3m/12m returns across assets
  6. Methodology — plain-English explanation + method picker

History doesn't repeat, but it rhymes.
"""

from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rhyme_lib.backtest import (
    backtest_stats,
    format_backtest,
    format_walk_forward,
    walk_forward_backtest,
)
from rhyme_lib.features import build_window_features
from rhyme_lib.forward_returns import DEFAULT_ASSETS, DEFAULT_HORIZONS_WEEKS, forward_returns
from rhyme_lib.labeler import label_clusters, label_map
from rhyme_lib.panel import (
    DEFAULT_BUCKETS,
    DEFAULT_SPECS,
    DEFAULT_TRANSFORMS,
    load_default_panel,
)
from rhyme_lib.similarity import (
    cosine_kmeans_similarity,
    embed_2d,
    gmm_similarity,
    primary_similarity,
    secondary_similarity,
)
from rhyme_lib.transforms import (
    infer_transforms,
    resample_panel,
    theme_aggregate,
    transform_and_zscore,
)

st.set_page_config(page_title="Rhyme", layout="wide", page_icon="📈")

NBER_RECESSIONS: list[tuple[str, str]] = [
    ("1957-08-01", "1958-04-30"),
    ("1960-04-01", "1961-02-28"),
    ("1969-12-01", "1970-11-30"),
    ("1973-11-01", "1975-03-31"),
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

# Big-ticket macro / market events — annotated on the Cycle scatter so the
# chart reads as history, not just a blob of dots.
CYCLE_EVENTS: list[tuple[str, str]] = [
    ("1973-11-30", "OPEC oil shock"),
    ("1980-03-31", "Volcker peak"),
    ("1987-10-31", "Black Monday"),
    ("2000-03-31", "Dot-com peak"),
    ("2008-09-30", "Lehman / GFC"),
    ("2011-08-31", "US debt downgrade"),
    ("2020-03-31", "COVID trough"),
    ("2022-06-30", "2022 inflation peak"),
]

BUCKET_LABELS = {
    "growth": "Growth",
    "inflation": "Inflation",
    "monetary": "Monetary / Financial",
    "sentiment": "Sentiment",
}


# ---------------------------------------------------------------------------
# Data loading with cache
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_default_panel():
    panel, meta = load_default_panel()
    return panel, meta


@st.cache_data(show_spinner=False)
def _cached_pipeline(
    panel_key: str,
    panel_bytes: bytes,
    freq: str,
    zmode: str,
    rolling_years: int,
    min_years: int,
    window_size: int,
    feature_codes: tuple[str, ...],
    n_clusters: int,
    method: str,
    robust: bool,
    label_mode: str,
    transforms_key: str,
    buckets_key: str,
):
    """Key-based cache wrapper so Streamlit only re-runs when inputs change.

    z and themes are computed over the FULL panel (so labeler always has
    growth/inflation/monetary). feature_codes narrows the series fed into
    the window feature vector — this is how the Macro/Market toggle works.
    `robust=True` swaps mean→median / std→MAD / Pearson→Spearman and uses
    MinCovDet in the Mahalanobis method.
    """
    panel = pd.read_parquet(io.BytesIO(panel_bytes))
    panel.index = pd.to_datetime(panel.index)
    transforms = dict(_decode_pairs(transforms_key))
    buckets = dict(_decode_pairs(buckets_key))

    z = transform_and_zscore(
        panel, transforms, freq=freq, mode=zmode,
        rolling_years=rolling_years, min_years=min_years, robust=robust,
    )
    themes = theme_aggregate(z, buckets, robust=robust)

    z_cols = [f"{c}_z" for c in feature_codes if f"{c}_z" in z.columns]
    z_for_features = z[z_cols]
    wf = build_window_features(z_for_features, window_size=window_size, n_pca=3, robust=robust)

    # Allow partial window overlap when history is short (common for macro +
    # 60-month window where z-valid rows only start ~2018).
    min_gap = max(window_size // 4, 6)
    if method == "primary":
        res = primary_similarity(
            wf, n_clusters=n_clusters, top_k=20, min_gap=min_gap, robust=robust,
        )
    elif method == "secondary":
        res = secondary_similarity(wf, top_k=20, min_gap=min_gap)
    elif method == "cosine":
        res = cosine_kmeans_similarity(
            wf, n_clusters=n_clusters, top_k=20, min_gap=min_gap, robust=robust,
        )
    elif method == "gmm":
        res = gmm_similarity(
            wf, n_clusters=n_clusters, top_k=20, min_gap=min_gap, robust=robust,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    rlabels = label_clusters(
        themes, res.labels, wf.end_dates,
        mode=label_mode, robust=robust, individual_z=z,
    )
    return {
        "panel": panel,
        "z": z,
        "themes": themes,
        "wf_features": wf.features,
        "wf_end_dates": wf.end_dates,
        "wf_panel_slice": wf.panel_slice,
        "wf_window_size": wf.window_size,
        "sim_labels": res.labels,
        "sim_distances": res.distances,
        "sim_reference_idx": res.reference_idx,
        "sim_top_analogs": res.top_analogs,
        "sim_method": res.method,
        "regime_labels": [asdict(r) for r in rlabels],
    }


@st.cache_data(show_spinner=False)
def _cached_walk_forward(
    panel_key: str,
    panel_bytes: bytes,
    freq: str,
    method: str,
    robust: bool,
    top_k: int,
    horizon_key: str,
    window_size: int,
    feature_codes_key: str,
    zmode: str,
    rolling_years: int,
    min_years: int,
    transforms_key: str,
    buckets_key: str,
):
    """Walk-forward backtest cache. Signature mirrors _cached_pipeline so we
    only recompute when something actually changed. We rebuild the WindowFeatures
    here rather than serialize the existing result dict — the numpy arrays
    aren't trivially cacheable as kwargs."""
    panel = pd.read_parquet(io.BytesIO(panel_bytes))
    panel.index = pd.to_datetime(panel.index)
    transforms_d = dict(_decode_pairs(transforms_key))
    buckets_d = dict(_decode_pairs(buckets_key))
    codes = tuple(feature_codes_key.split(",")) if feature_codes_key else ()

    z_local = transform_and_zscore(
        panel, transforms_d, freq=freq, mode=zmode,
        rolling_years=rolling_years, min_years=min_years, robust=robust,
    )
    z_cols = [f"{c}_z" for c in codes if f"{c}_z" in z_local.columns]
    wf = build_window_features(z_local[z_cols], window_size=window_size, n_pca=3, robust=robust)

    return walk_forward_backtest(
        wf_features=wf.features,
        wf_end_dates=wf.end_dates,
        wf_panel_slice=wf.panel_slice,
        window_size=wf.window_size,
        panel_daily=panel,
        freq=freq,
        method=method,
        robust=robust,
        top_k=top_k,
        horizon_key=horizon_key,
    )


def _encode_pairs(d: dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(d.items()))


def _decode_pairs(s: str) -> list[tuple[str, str]]:
    if not s:
        return []
    return [tuple(p.split("=", 1)) for p in s.split("|") if "=" in p]


# ---------------------------------------------------------------------------
# Sidebar — data source + all parameters
# ---------------------------------------------------------------------------


st.sidebar.title("Rhyme")
st.sidebar.caption("History doesn't repeat, but it rhymes.")

data_choice = st.sidebar.radio(
    "Panel source",
    ["Default (built-in)", "Upload CSV or JSON"],
    horizontal=False,
)

uploaded = None
if data_choice == "Upload CSV or JSON":
    uploaded = st.sidebar.file_uploader("File (first column = date)", type=["csv", "json"])

mode = st.sidebar.radio(
    "Mode",
    ["Macro", "Market"],
    index=0,
    horizontal=True,
    help="Macro = growth + inflation series @ monthly, 60-month window. "
         "Market = monetary + sentiment series @ weekly, 152-week window.",
)

MODE_BUCKETS = {
    "Macro": {"growth", "inflation"},
    "Market": {"monetary", "sentiment"},
}
MODE_DEFAULT_FREQ = {"Macro": "M", "Market": "W"}
MODE_DEFAULT_WINDOW = {"Macro": 60, "Market": 152}
MODE_WINDOW_RANGE = {"Macro": (12, 120), "Market": (26, 260)}

bucket_selection = MODE_BUCKETS[mode]

with st.sidebar.expander("Frequency & transformation", expanded=False):
    freq = st.radio(
        "Target frequency", ["W", "M"],
        index=0 if MODE_DEFAULT_FREQ[mode] == "W" else 1,
        horizontal=True,
        help="Weekly for markets; monthly for macro. Default panel ships at daily and is resampled.",
    )
    zmode = st.radio("Z-score window", ["rolling", "expanding"], index=0, horizontal=True)
    rolling_years = st.slider("Rolling years", 5, 40, 20, step=5)
    min_years = st.slider("Minimum history (years)", 2, 15, 10, step=1)
    robust = st.checkbox(
        "Robust mode",
        value=False,
        help=(
            "Winsorize inputs, z-score with median + MAD instead of mean + std, "
            "use Spearman (rank) cross-series correlations, take median over "
            "cluster members for regime labels, and swap Ledoit-Wolf for "
            "Min-Covariance-Determinant in the Mahalanobis engine. Slower "
            "but much less sensitive to COVID / GFC outliers."
        ),
    )

with st.sidebar.expander("Windowing & clustering", expanded=True):
    wmin, wmax = MODE_WINDOW_RANGE[mode]
    wdef = MODE_DEFAULT_WINDOW[mode]
    unit = "months" if freq == "M" else "weeks"
    window_size = st.slider(
        f"Window size ({unit})", wmin, wmax, wdef,
        help=f"{wdef} {unit} is the default for {mode} mode.",
    )
    n_clusters = st.slider("Number of regime clusters", 2, 10, 4)

METHOD_LABELS = {
    "primary":   "Primary: Mahalanobis + Ward",
    "secondary": "Secondary: SBD + HDBSCAN",
    "cosine":    "Cosine + KMeans (direction-based)",
    "gmm":       "Euclidean + GMM (soft clusters)",
}
method = st.sidebar.radio(
    "Similarity methodology",
    list(METHOD_LABELS.keys()),
    format_func=lambda x: METHOD_LABELS[x],
    index=0,
)

# ---------------------------------------------------------------------------
# Build panel
# ---------------------------------------------------------------------------


def _load_upload(file) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    name = getattr(file, "name", "").lower()
    raw = pd.read_json(file) if name.endswith(".json") else pd.read_csv(file)
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.set_index(date_col).sort_index().apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(how="all")
    transforms = infer_transforms(raw)
    buckets = {c: "monetary" for c in raw.columns}
    return raw, transforms, buckets


if data_choice == "Default (built-in)":
    try:
        panel, meta = _cached_default_panel()
    except FileNotFoundError as e:
        st.error(
            f"{e}\n\nRun `python refresh_panel.py` locally (requires FRED_API_KEY) "
            "to build the cache, then redeploy."
        )
        st.stop()
    all_codes = [s.code for s in DEFAULT_SPECS if s.code in panel.columns]
    feature_codes = [c for c in all_codes if DEFAULT_BUCKETS.get(c) in bucket_selection]
    transforms = DEFAULT_TRANSFORMS
    buckets = DEFAULT_BUCKETS
    panel_source_name = "default"
    # Drop series from the feature set (but keep in panel/themes) when they
    # don't extend near the panel's current edge or don't cover enough history
    # for a reasonable window. `dropna(how="any")` in feature building means
    # a single stale/short series otherwise collapses the whole matrix.
    _panel_end = pd.to_datetime(panel.index.max())
    _stale_cutoff = _panel_end - pd.Timedelta(days=400)
    _short_cutoff = _panel_end - pd.Timedelta(days=365 * 10)
    _excluded = []
    _kept = []
    for c in feature_codes:
        s = panel[c].dropna()
        if s.empty or s.index.max() < _stale_cutoff or s.index.min() > _short_cutoff:
            _excluded.append(c)
        else:
            _kept.append(c)
    if _excluded:
        st.sidebar.caption(
            f"Excluded from features (kept in themes): {', '.join(_excluded)}"
        )
    feature_codes = _kept
else:
    if uploaded is None:
        st.info("Upload a CSV or JSON in the sidebar (first column = date, rest numeric).")
        st.stop()
    panel, transforms, buckets = _load_upload(uploaded)
    meta = pd.DataFrame(
        [
            {
                "code": c,
                "bucket": buckets[c],
                "transform": transforms[c],
                "source": "uploaded",
                "n_obs": int(panel[c].notna().sum()),
                "start": panel[c].dropna().index.min() if panel[c].notna().any() else None,
                "end":   panel[c].dropna().index.max() if panel[c].notna().any() else None,
            }
            for c in panel.columns
        ]
    )
    all_codes = list(panel.columns)
    feature_codes = list(panel.columns)
    panel_source_name = getattr(uploaded, "name", "uploaded")

if not feature_codes:
    st.warning(f"No series in the {mode} buckets — check your panel bucket tags.")
    st.stop()

# Cache-stable serialization of FULL panel (themes need all 4 buckets for labeling)
_panel_bytes = io.BytesIO()
panel[all_codes].to_parquet(_panel_bytes)
_panel_bytes.seek(0)

try:
    result = _cached_pipeline(
        panel_key=panel_source_name,
        panel_bytes=_panel_bytes.getvalue(),
        freq=freq,
        zmode=zmode,
        rolling_years=rolling_years,
        min_years=min_years,
        window_size=window_size,
        feature_codes=tuple(feature_codes),
        n_clusters=n_clusters,
        method=method,
        robust=robust,
        label_mode=("market" if mode == "Market" else "macro"),
        transforms_key=_encode_pairs({k: v for k, v in transforms.items() if k in all_codes}),
        buckets_key=_encode_pairs({k: v for k, v in buckets.items() if k in all_codes}),
    )
except ValueError as e:
    st.error(f"Pipeline failed: {e}\n\nTry a shorter window or different frequency.")
    st.stop()

regime_labels = result["regime_labels"]
lm = {rl["cluster"]: rl["label"] for rl in regime_labels}
ref_idx = result["sim_reference_idx"]
ref_end = result["wf_end_dates"][ref_idx]
ref_cluster = int(result["sim_labels"][ref_idx])
ref_label = lm.get(ref_cluster, f"#{ref_cluster}")
top_analogs = result["sim_top_analogs"].copy()
top_analogs["label"] = top_analogs["cluster"].map(lm)

# Forward returns off the daily panel (use original unfiltered panel to reach all assets)
fwd_by_horizon = {
    hname: forward_returns(panel, top_analogs["end_date"], horizon_weeks=h)
    for hname, h in DEFAULT_HORIZONS_WEEKS.items()
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


tab_overview, tab_data, tab_cycle, tab_map, tab_ts, tab_analogs, tab_method = st.tabs(
    ["Overview", "Data", "Cycle", "Regime map", "Time series", "Analogs", "Methodology"]
)

# --- Overview --------------------------------------------------------------

with tab_overview:
    st.title("Rhyme")
    st.caption("Historical analog / regime detection on US macro + market panels.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference window ends", ref_end.strftime("%Y-%m-%d"))
    col2.metric("Today's regime", ref_label)
    col3.metric("Windows analyzed", f"{len(result['wf_end_dates']):,}")
    col4.metric("Method", METHOD_LABELS[method].split(":", 1)[-1].strip())

    st.subheader("Top 5 analogs")
    compact = top_analogs.head(5).copy()
    compact["window_start"] = compact["end_date"] - pd.Timedelta(weeks=window_size if freq == "W" else window_size * 4)
    display = compact[["rank", "window_start", "end_date", "distance", "label"]].copy()
    display["window_start"] = display["window_start"].dt.strftime("%Y-%m-%d")
    display["end_date"] = display["end_date"].dt.strftime("%Y-%m-%d")
    display["distance"] = display["distance"].apply(lambda v: f"{v:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Theme z-scores (last 4 observations)")
    st.dataframe(result["themes"].dropna(how="all").tail(4).round(2), use_container_width=True)

# --- Data ------------------------------------------------------------------

with tab_data:
    st.subheader(f"Panel ({panel_source_name})")
    st.write(
        f"{panel.shape[0]:,} rows × {panel.shape[1]} series  "
        f"|  {panel.index.min().date()} → {panel.index.max().date()}  "
        f"|  {mode} mode feeds {len(feature_codes)} series into the similarity engine"
    )
    show_meta = meta.copy()
    if "start" in show_meta.columns and show_meta["start"].notna().any():
        show_meta["start"] = pd.to_datetime(show_meta["start"]).dt.date
    if "end" in show_meta.columns and show_meta["end"].notna().any():
        show_meta["end"] = pd.to_datetime(show_meta["end"]).dt.date
    st.dataframe(show_meta, use_container_width=True, hide_index=True)

    st.subheader("Panel tail")
    st.dataframe(panel[feature_codes].tail(10).round(3), use_container_width=True)

# --- Cycle (three clocks) --------------------------------------------------


def _clock_df(source: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return source[[x_col, y_col]].dropna()


def _draw_clock(
    cycle_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    quad_labels: dict[str, str],
    quad_colors: dict[str, str],
    events: list[tuple[str, str]],
    freq: str,
    title_note: str,
):
    offset_map = {"M": {"1M": 1, "3M": 3, "6M": 6, "12M": 12},
                  "W": {"1M": 4, "3M": 13, "6M": 26, "12M": 52},
                  "D": {"1M": 21, "3M": 63, "6M": 126, "12M": 252}}
    offsets = offset_map.get(freq, offset_map["M"])
    today_pos = len(cycle_df) - 1

    trail_points = []
    for label, back in [("T-12M", offsets["12M"]), ("T-6M", offsets["6M"]),
                        ("T-3M", offsets["3M"]), ("T-1M", offsets["1M"]),
                        ("Today", 0)]:
        pos = today_pos - back
        if pos >= 0:
            row = cycle_df.iloc[pos]
            trail_points.append({
                "label": label,
                "date": cycle_df.index[pos],
                "x": float(row[x_col]),
                "y": float(row[y_col]),
            })
    trail = pd.DataFrame(trail_points)

    xr = float(max(1.5, cycle_df[x_col].abs().quantile(0.98) + 0.4))
    yr = float(max(1.5, cycle_df[y_col].abs().quantile(0.98) + 0.4))

    fig = go.Figure()
    for quad, (x0, x1, y0, y1) in {
        "NE": (0, xr, 0, yr), "NW": (-xr, 0, 0, yr),
        "SE": (0, xr, -yr, 0), "SW": (-xr, 0, -yr, 0),
    }.items():
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=quad_colors[quad], line=dict(width=0), layer="below")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=quad_labels[quad], showarrow=False,
            font=dict(size=11, color="rgba(180,180,180,0.55)"),
        )

    fig.add_shape(type="line", x0=-xr, x1=xr, y0=0, y1=0,
                  line=dict(color="rgba(180,180,180,0.5)", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-yr, y1=yr,
                  line=dict(color="rgba(180,180,180,0.5)", width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=cycle_df[x_col], y=cycle_df[y_col],
        mode="markers",
        marker=dict(size=4, color="rgba(170,170,170,0.35)"),
        hovertemplate="%{customdata|%Y-%m-%d}<br>"
                      f"{x_title}=%{{x:.2f}}  {y_title}=%{{y:.2f}}<extra>history</extra>",
        customdata=cycle_df.index,
        name="history",
    ))

    for date_str, label in events:
        d = pd.Timestamp(date_str)
        if d not in cycle_df.index:
            nearest = cycle_df.index[cycle_df.index.get_indexer([d], method="nearest")[0]]
            d = nearest
        if d in cycle_df.index:
            row = cycle_df.loc[d]
            fig.add_trace(go.Scatter(
                x=[row[x_col]], y=[row[y_col]],
                mode="markers+text",
                marker=dict(size=9, color="rgba(60,60,60,0.85)", symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f" {label}"], textposition="top right",
                textfont=dict(size=10, color="rgba(60,60,60,0.9)"),
                hovertemplate=f"{label}<br>{d.date()}<extra></extra>",
                showlegend=False,
            ))

    if not trail.empty:
        fig.add_trace(go.Scatter(
            x=trail["x"], y=trail["y"],
            mode="lines+markers+text",
            line=dict(color="#FFD700", width=2.5),
            marker=dict(size=[8, 10, 12, 14, 20],
                        color=["#FFB800", "#FFC400", "#FFD000", "#FFDC00", "#FFD700"],
                        symbol=["circle"] * 4 + ["star"],
                        line=dict(color="#D10000", width=2)),
            text=trail["label"],
            textposition="bottom center",
            textfont=dict(size=11, color="#FFD700"),
            hovertemplate="%{text}<br>%{customdata|%Y-%m-%d}<br>"
                          f"{x_title}=%{{x:.2f}}  {y_title}=%{{y:.2f}}<extra></extra>",
            customdata=trail["date"],
            name="recent trail",
        ))

    fig.update_layout(
        height=620,
        xaxis=dict(title=x_title, range=[-xr, xr], zeroline=False),
        yaxis=dict(title=y_title, range=[-yr, yr], zeroline=False),
        showlegend=False,
        hovermode="closest",
    )
    return fig, trail


with tab_cycle:
    clock = st.radio(
        "Clock view",
        ["Macro (growth × inflation)",
         "Market (vol × valuation)",
         "Sentiment (sentiment × stress)"],
        index=0,
        horizontal=True,
        help=(
            "Three different ways to locate where we are today. "
            "**Macro** is the classic Merrill Lynch investment clock. "
            "**Market** plots VIX (vol) against credit spreads (valuation). "
            "**Sentiment** plots UMich consumer sentiment against an "
            "aggregate financial-stress axis."
        ),
    )

    themes_all = result["themes"].dropna(how="all")
    z_all = result["z"]

    if clock.startswith("Macro"):
        x_col, y_col = "growth", "inflation"
        source = themes_all
        x_title, y_title = "Growth z-score", "Inflation z-score"
        quad_labels = {
            "NE": "Reflation (high G, high I)",
            "NW": "Stagflation (low G, high I)",
            "SE": "Goldilocks (high G, low I)",
            "SW": "Deflationary bust (low G, low I)",
        }
        quad_colors = {"NE": "rgba(255,170,60,0.10)",
                       "NW": "rgba(220,60,60,0.10)",
                       "SE": "rgba(60,200,100,0.10)",
                       "SW": "rgba(60,120,220,0.10)"}
        note = (
            "Each dot is one observation on the Merrill Lynch-style clock. "
            "Upper-right = Reflation; upper-left = Stagflation; lower-right = "
            "Goldilocks; lower-left = Deflationary bust."
        )
    elif clock.startswith("Market"):
        # Valuation axis: mean of baa & aaa credit spread z (higher = wider = cheaper = stressed).
        spreads = [c for c in ("baa_spread_z", "aaa_spread_z") if c in z_all.columns]
        if "vix_z" not in z_all.columns or not spreads:
            st.info("Need vix_z and credit spread z in the z panel for the market clock.")
            st.stop()
        vix_s = z_all["vix_z"]
        val_s = z_all[spreads].mean(axis=1)
        source = pd.DataFrame({"vix": vix_s, "valuation": val_s}).dropna()
        x_col, y_col = "vix", "valuation"
        x_title, y_title = "VIX z (vol)", "Credit spread z (valuation)"
        quad_labels = {
            "NE": "Panic (high vol, cheap credit)",
            "NW": "Calm but cheap (accumulate)",
            "SE": "Topping (high vol, rich credit)",
            "SW": "Melt-up / complacency",
        }
        quad_colors = {"NE": "rgba(220,60,60,0.12)",
                       "NW": "rgba(60,200,100,0.12)",
                       "SE": "rgba(255,170,60,0.12)",
                       "SW": "rgba(60,120,220,0.10)"}
        note = (
            "X = VIX z (how scared options are); Y = mean credit-spread z "
            "(how cheap / distressed credit is). Upper-right = Panic; "
            "lower-left = Melt-up / complacency; lower-right = Topping "
            "(vol rising before spreads widen); upper-left = calm-but-cheap "
            "is rare and usually a bargain."
        )
    else:  # Sentiment
        if "umich_sentiment_z" not in z_all.columns:
            st.info("Need umich_sentiment_z for the sentiment clock.")
            st.stop()
        stress_pool = [c for c in ("nfci_z", "vix_z", "baa_spread_z") if c in z_all.columns]
        if not stress_pool:
            st.info("Need at least one stress series (nfci / vix / baa_spread) for the sentiment clock.")
            st.stop()
        sent_s = z_all["umich_sentiment_z"]
        # Mean across available stress proxies — higher = more stress.
        stress_s = z_all[stress_pool].mean(axis=1)
        source = pd.DataFrame({"sentiment": sent_s, "stress": stress_s}).dropna()
        x_col, y_col = "sentiment", "stress"
        x_title, y_title = "Consumer sentiment z", "Financial stress z"
        quad_labels = {
            "NE": "Disbelief (bullish + stressed)",
            "NW": "Fear (bearish + stressed)",
            "SE": "Euphoria (bullish + calm)",
            "SW": "Apathy (bearish + calm)",
        }
        quad_colors = {"NE": "rgba(255,170,60,0.10)",
                       "NW": "rgba(220,60,60,0.12)",
                       "SE": "rgba(60,200,100,0.10)",
                       "SW": "rgba(150,150,150,0.10)"}
        note = (
            "X = UMich consumer sentiment z. Y = composite financial-stress "
            "z (NFCI / VIX / Baa spread). Upper-right = Disbelief (people "
            "still bullish but markets nervous — often near a top); "
            "lower-right = Euphoria (bullish + calm — classic complacency); "
            "upper-left = Fear; lower-left = Apathy (bearish but calm — "
            "often near a bottom after capitulation)."
        )

    cycle_df = _clock_df(source, x_col, y_col)
    if len(cycle_df) < 13:
        st.info(f"Not enough history for the {clock.lower()}.")
    else:
        fig, trail = _draw_clock(
            cycle_df, x_col, y_col, x_title, y_title,
            quad_labels, quad_colors, CYCLE_EVENTS, freq, note,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not trail.empty:
            cap = trail.iloc[-1]
            st.caption(
                f"**Today ({cap['date'].date()}):** {x_title} = {cap['x']:+.2f}, "
                f"{y_title} = {cap['y']:+.2f}.  {note}  "
                f"Grey dots = every historical period. Diamonds = landmark events. "
                f"Gold trail = T-12M → T-6M → T-3M → T-1M → today."
            )

# --- Regime map (2D embedding) ---------------------------------------------

with tab_map:
    st.subheader("Regime map")
    embed_method = st.radio(
        "Embedding",
        ["pca", "umap", "tsne"],
        index=0,
        horizontal=True,
        help="PCA is linear, deterministic, and inspectable (default). UMAP preserves global structure. t-SNE looks nicer in 2D but distances between clusters are not meaningful.",
    )
    wf_obj = type("WF", (), {})()
    wf_obj.features = result["wf_features"]
    wf_obj.end_dates = result["wf_end_dates"]
    wf_obj.panel_slice = result["wf_panel_slice"]
    wf_obj.window_size = result["wf_window_size"]
    wf_obj.feature_names = None  # not used by embed_2d

    coords = embed_2d(wf_obj, method=embed_method, random_state=0)
    map_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": [lm.get(int(c), f"#{c}") for c in result["sim_labels"]],
        "end_date": result["wf_end_dates"],
    })
    fig = px.scatter(
        map_df, x="x", y="y", color="cluster",
        hover_data={"end_date": True, "x": ":.2f", "y": ":.2f"},
        opacity=0.55,
    )
    fig.add_trace(go.Scatter(
        x=[coords[ref_idx, 0]], y=[coords[ref_idx, 1]],
        mode="markers+text", text=["today"], textposition="top center",
        marker=dict(
            size=22, color="#FFD700", symbol="star",
            line=dict(color="#D10000", width=2),
        ),
        textfont=dict(color="#D10000", size=13),
        name="reference",
    ))
    fig.update_layout(height=560, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    if embed_method == "tsne":
        st.caption(
            "⚠️ t-SNE warps distances to preserve local neighborhoods. Distances "
            "between visible clusters are not meaningful — inspect individual "
            "dot placement, not gaps."
        )

# --- Time series -----------------------------------------------------------

with tab_ts:
    st.subheader("Theme z-scores over time")
    themes = result["themes"].dropna(how="all")
    fig = go.Figure()
    for col in themes.columns:
        fig.add_trace(go.Scatter(
            x=themes.index, y=themes[col], mode="lines",
            name=BUCKET_LABELS.get(col, col), connectgaps=False,
        ))

    end_dates = result["wf_end_dates"]
    labels = result["sim_labels"]
    cluster_palette = px.colors.qualitative.Set2
    unique = sorted(set(int(c) for c in labels))
    color_map_cluster = {c: cluster_palette[i % len(cluster_palette)] for i, c in enumerate(unique)}

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if len(end_dates) > 0:
        start = end_dates[0]; cur = int(labels[0])
        for i in range(1, len(end_dates)):
            if int(labels[i]) != cur:
                runs.append((start, end_dates[i - 1], cur))
                start = end_dates[i]; cur = int(labels[i])
        runs.append((start, end_dates[-1], cur))

    for s, e, c in runs:
        fig.add_vrect(
            x0=s, x1=e, fillcolor=color_map_cluster[c], opacity=0.12,
            line_width=0, layer="below",
        )

    for r_start, r_end in NBER_RECESSIONS:
        fig.add_vrect(
            x0=r_start, x1=r_end, fillcolor="gray", opacity=0.18,
            line_width=0, layer="below",
        )

    x_min = themes.index.min()
    x_max = themes.index.max()
    fig.update_layout(
        height=520, hovermode="x unified",
        yaxis_title="z-score", xaxis_title=None,
        xaxis=dict(range=[x_min, x_max]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Colored shading = regime cluster. Gray bands = NBER recessions."
    )

    with st.expander("Cluster summary", expanded=False):
        cluster_df = pd.DataFrame(regime_labels)
        if not cluster_df.empty:
            if mode == "Market":
                cols = ["cluster", "label", "n_windows",
                        "financial_z", "sentiment_z", "vix_z",
                        "growth_z", "inflation_z"]
            else:
                cols = ["cluster", "label", "n_windows",
                        "growth_z", "inflation_z", "financial_z",
                        "sentiment_z"]
            cluster_df = cluster_df[cols]
            num_cols = [c for c in cols if c.endswith("_z")]
            cluster_df[num_cols] = cluster_df[num_cols].round(2)
            cluster_df = cluster_df.rename(columns={"financial_z": "monetary_z"})
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)

# --- Analogs ---------------------------------------------------------------

with tab_analogs:
    st.subheader(f"Top analogs vs. {ref_end.strftime('%Y-%m-%d')}")
    c1, c2 = st.columns([1, 1])
    with c1:
        horizon = st.radio(
            "Forward horizon",
            list(DEFAULT_HORIZONS_WEEKS.keys()), index=1, horizontal=True,
        )
    with c2:
        top_k = st.selectbox("Show top K", [5, 10, 15, 20], index=1)

    fwd = fwd_by_horizon[horizon].copy()
    fwd.index = top_analogs["end_date"].values

    base = top_analogs.head(top_k).copy()
    fwd_k = fwd.iloc[:top_k].copy()
    base["window_start"] = base["end_date"] - pd.Timedelta(
        weeks=window_size if freq == "W" else window_size * 4
    )
    base = base[["rank", "window_start", "end_date", "distance", "cluster", "label"]]

    ret_cols = [a for a, k in DEFAULT_ASSETS.items() if k == "ret" and a in fwd_k.columns]
    bps_cols = [a for a, k in DEFAULT_ASSETS.items() if k == "bps" and a in fwd_k.columns]

    # Average row — computed on RAW numeric forward returns (before formatting).
    avg_row = {
        "rank": "Avg",
        "window_start": "",
        "end_date": "",
        "distance": base["distance"].mean(),
        "cluster": "",
        "label": f"Average of top {top_k}",
    }
    for c in fwd_k.columns:
        avg_row[c] = fwd_k[c].mean(skipna=True)

    # Build combined frame (avg first, then the K rows) with numeric forward cols.
    numeric = pd.concat(
        [base.reset_index(drop=True), fwd_k.reset_index(drop=True)],
        axis=1,
    )
    numeric = pd.concat([pd.DataFrame([avg_row]), numeric], ignore_index=True)

    # Format for display
    disp = numeric.copy()
    disp["window_start"] = disp["window_start"].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x
    )
    disp["end_date"] = disp["end_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x
    )
    disp["distance"] = disp["distance"].apply(
        lambda v: f"{v:.3f}" if pd.notna(v) else ""
    )
    for c in ret_cols:
        disp[c] = numeric[c].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else ""
        )
    for c in bps_cols:
        disp[c] = numeric[c].apply(
            lambda v: f"{int(round(v))} bps" if pd.notna(v) else ""
        )

    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        "`spx`, `dxy`, `gold`, `wti` = forward log returns (shown as %). "
        "`ust10_yield`, `baa_spread`, `aaa_spread` = forward changes in basis points. "
        "The **Avg** row averages the top K forward outcomes."
    )

    st.subheader("Series overlay: reference window vs. top analogs")
    series_options = [c for c in result["z"].columns if c.endswith("_z")]
    pick = st.selectbox("Series", series_options, index=0)
    series_raw_col = pick[:-2]
    zpanel = result["z"]
    ref_block = zpanel[pick].iloc[
        zpanel.index.get_loc(ref_end) - window_size + 1 : zpanel.index.get_loc(ref_end) + 1
    ].reset_index(drop=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=ref_block.values, name="reference (today)",
        line=dict(width=4, color="#FFD700"),
    ))
    for _, row in top_analogs.head(5).iterrows():
        end = row["end_date"]
        if end not in zpanel.index:
            continue
        pos = zpanel.index.get_loc(end)
        start = pos - window_size + 1
        blk = zpanel[pick].iloc[start : pos + 1].reset_index(drop=True)
        fig2.add_trace(go.Scatter(y=blk.values, name=str(end.date()), opacity=0.55))
    fig2.update_layout(height=420, yaxis_title=f"{series_raw_col} (z)", xaxis_title="periods into window")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Cross-sectional: dispersion of today's K analog outcomes ----------
    st.markdown("---")
    st.subheader("Analog consensus — distribution of today's top K outcomes")
    st.caption(
        f"The forward returns (or yield / spread changes) from the top {top_k} "
        f"analogs at the **{horizon}** horizon are treated as the distribution "
        f"of possible outcomes if today rhymes with those regimes. "
        f"**This is not a strategy Sharpe** — it only describes how tightly the "
        f"analogs agree. For honest return / vol / drawdown numbers see the "
        f"walk-forward backtest below."
    )

    stats = backtest_stats(fwd_k, horizon_key=horizon)
    if stats.empty:
        st.info("No valid forward outcomes for this horizon.")
    else:
        formatted = format_backtest(stats)
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        dist_asset = st.selectbox(
            "Distribution plot — asset",
            list(stats["asset"]),
            index=0,
        )
        kind = stats.loc[stats["asset"] == dist_asset, "kind"].iloc[0]
        values = fwd_k[dist_asset].dropna()
        if kind == "ret":
            vals_display = values * 100.0
            unit = "%"
        else:
            vals_display = values
            unit = "bps"

        fig3 = go.Figure()
        fig3.add_trace(go.Box(
            x=vals_display, boxpoints="all", jitter=0.6, pointpos=0,
            marker=dict(color="#FFD700", size=9, line=dict(color="#D10000", width=1)),
            line=dict(color="#888"), fillcolor="rgba(255,215,0,0.12)",
            name=dist_asset, orientation="h",
        ))
        fig3.add_vline(x=0, line_width=1, line_color="rgba(180,180,180,0.5)",
                       line_dash="dot")
        mean_v = float(vals_display.mean())
        fig3.add_vline(x=mean_v, line_width=2, line_color="#D10000",
                       annotation_text=f"mean {mean_v:.2f}{unit}",
                       annotation_position="top")
        fig3.update_layout(
            height=280, xaxis_title=f"{dist_asset} forward ({unit})",
            yaxis=dict(showticklabels=False), showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Agree = fraction of analogs with favorable outcome "
            "(positive for price assets; tightening — i.e. falling rate or spread — "
            "for yield and spread assets)."
        )

    # --- Walk-forward backtest (no look-ahead) -----------------------------
    st.markdown("---")
    st.subheader("Walk-forward backtest — no look-ahead")
    st.caption(
        f"At every historical decision date **T**, we rebuild the similarity "
        f"engine using only features from strictly before T − {horizon} (so every "
        f"candidate analog's forward return has already been realized). The "
        f"position is **sign(mean forward return of the top {top_k} past analogs)**; "
        f"the strategy return is position × realized(T → T + {horizon}). Trades are "
        f"non-overlapping (step = horizon). Early history uses fewer analogs "
        f"because fewer past windows exist — that's intentional. "
        f"Sharpe is annualized ({int(round(12 if horizon == '1m' else 4 if horizon == '3m' else 1))} trades/year)."
    )

    run_wf = st.checkbox(
        "Run walk-forward backtest (slower — re-fits the engine at every T)",
        value=False,
    )
    if run_wf:
        with st.spinner("Running walk-forward… re-fitting the engine at every historical T"):
            try:
                wf_result = _cached_walk_forward(
                    panel_key=panel_source_name,
                    panel_bytes=_panel_bytes.getvalue(),
                    freq=freq,
                    method=method,
                    robust=robust,
                    top_k=top_k,
                    horizon_key=horizon,
                    window_size=window_size,
                    feature_codes_key=",".join(feature_codes),
                    zmode=zmode,
                    rolling_years=rolling_years,
                    min_years=min_years,
                    transforms_key=_encode_pairs(
                        {k: v for k, v in transforms.items() if k in all_codes}
                    ),
                    buckets_key=_encode_pairs(
                        {k: v for k, v in buckets.items() if k in all_codes}
                    ),
                )
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")
                wf_result = None

        if wf_result is not None:
            wf_stats = wf_result["stats"]
            if wf_stats.empty:
                st.info(
                    "Not enough past history for a walk-forward backtest at this "
                    "horizon / window combo — widen rolling history or shrink the "
                    "window."
                )
            else:
                st.dataframe(
                    format_walk_forward(wf_stats),
                    use_container_width=True,
                    hide_index=True,
                )

                trades = wf_result["trades"]
                equity = wf_result["equity"]
                if len(trades) > 0 and equity:
                    st.markdown("**Equity curves** (cumulative, position-signed)")
                    eq_fig = go.Figure()
                    for asset, curve in equity.items():
                        kind = DEFAULT_ASSETS.get(asset, "ret")
                        y = (curve - 1.0) * 100.0 if kind == "ret" else curve
                        eq_fig.add_trace(go.Scatter(
                            x=curve.index, y=y, mode="lines", name=asset,
                        ))
                    eq_fig.add_hline(y=0, line_width=1,
                                     line_color="rgba(180,180,180,0.5)",
                                     line_dash="dot")
                    eq_fig.update_layout(
                        height=400,
                        yaxis_title="cumulative (% for prices, bps for yields/spreads)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(eq_fig, use_container_width=True)

                    with st.expander(f"Trade log ({len(trades)} trades)", expanded=False):
                        show = trades.copy()
                        show["date"] = pd.to_datetime(show["date"]).dt.date
                        st.dataframe(show, use_container_width=True, hide_index=True)

                first_trade = trades["date"].min()
                last_trade = trades["date"].max()
                st.caption(
                    f"{len(trades)} non-overlapping trades from "
                    f"{pd.Timestamp(first_trade).date()} to "
                    f"{pd.Timestamp(last_trade).date()}. "
                    f"Signal at each T uses only windows ending ≤ T − {horizon}; "
                    f"no data from after T is used to form the position."
                )

# --- Methodology -----------------------------------------------------------

with tab_method:
    st.subheader("Methodology")
    st.markdown(Path(__file__).parent.joinpath("METHODOLOGY.md").read_text())
