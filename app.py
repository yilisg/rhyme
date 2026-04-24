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

from rhyme_lib.features import build_window_features
from rhyme_lib.forward_returns import DEFAULT_ASSETS, DEFAULT_HORIZONS_WEEKS, forward_returns
from rhyme_lib.labeler import label_clusters, label_map
from rhyme_lib.panel import (
    DEFAULT_BUCKETS,
    DEFAULT_SPECS,
    DEFAULT_TRANSFORMS,
    load_default_panel,
)
from rhyme_lib.similarity import embed_2d, primary_similarity, secondary_similarity
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
    transforms_key: str,
    buckets_key: str,
):
    """Key-based cache wrapper so Streamlit only re-runs when inputs change.

    z and themes are computed over the FULL panel (so labeler always has
    growth/inflation/monetary). feature_codes narrows the series fed into
    the window feature vector — this is how the Macro/Market toggle works.
    """
    panel = pd.read_parquet(io.BytesIO(panel_bytes))
    panel.index = pd.to_datetime(panel.index)
    transforms = dict(_decode_pairs(transforms_key))
    buckets = dict(_decode_pairs(buckets_key))

    z = transform_and_zscore(
        panel, transforms, freq=freq, mode=zmode,
        rolling_years=rolling_years, min_years=min_years,
    )
    themes = theme_aggregate(z, buckets)

    z_cols = [f"{c}_z" for c in feature_codes if f"{c}_z" in z.columns]
    z_for_features = z[z_cols]
    wf = build_window_features(z_for_features, window_size=window_size, n_pca=3)

    # Allow partial window overlap when history is short (common for macro +
    # 60-month window where z-valid rows only start ~2018).
    min_gap = max(window_size // 4, 6)
    if method == "primary":
        res = primary_similarity(wf, n_clusters=n_clusters, top_k=20, min_gap=min_gap)
    else:
        res = secondary_similarity(wf, top_k=20, min_gap=min_gap)

    rlabels = label_clusters(themes, res.labels, wf.end_dates)
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

with st.sidebar.expander("Windowing & clustering", expanded=True):
    wmin, wmax = MODE_WINDOW_RANGE[mode]
    wdef = MODE_DEFAULT_WINDOW[mode]
    unit = "months" if freq == "M" else "weeks"
    window_size = st.slider(
        f"Window size ({unit})", wmin, wmax, wdef,
        help=f"{wdef} {unit} is the default for {mode} mode.",
    )
    n_clusters = st.slider("Number of regime clusters", 2, 10, 4)

method = st.sidebar.radio(
    "Similarity methodology",
    ["primary", "secondary"],
    format_func=lambda x: (
        "Primary: Mahalanobis + Ward" if x == "primary"
        else "Secondary: SBD + HDBSCAN"
    ),
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


tab_overview, tab_data, tab_map, tab_ts, tab_analogs, tab_method = st.tabs(
    ["Overview", "Data", "Regime map", "Time series", "Analogs", "Methodology"]
)

# --- Overview --------------------------------------------------------------

with tab_overview:
    st.title("Rhyme")
    st.caption("Historical analog / regime detection on US macro + market panels.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference window ends", ref_end.strftime("%Y-%m-%d"))
    col2.metric("Today's regime", ref_label)
    col3.metric("Windows analyzed", f"{len(result['wf_end_dates']):,}")
    col4.metric("Method", "Mahalanobis + Ward" if method == "primary" else "SBD + HDBSCAN")

    st.subheader("Top 5 analogs")
    compact = top_analogs.head(5).copy()
    compact["window_start"] = compact["end_date"] - pd.Timedelta(weeks=window_size if freq == "W" else window_size * 4)
    display = compact[["rank", "window_start", "end_date", "distance", "label"]].copy()
    display["window_start"] = display["window_start"].dt.strftime("%Y-%m-%d")
    display["end_date"] = display["end_date"].dt.strftime("%Y-%m-%d")
    display["distance"] = display["distance"].round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Theme z-scores (last 4 observations)")
    st.dataframe(result["themes"].dropna(how="all").tail(4).round(2), use_container_width=True)

# --- Data ------------------------------------------------------------------

with tab_data:
    st.subheader(f"Panel ({panel_source_name})")
    st.write(
        f"{panel.shape[0]:,} rows × {panel.shape[1]} series  "
        f"|  {panel.index.min().date()} → {panel.index.max().date()}  "
        f"|  selected: {len(selected_codes_all)} series"
    )
    show_meta = meta.copy()
    if "start" in show_meta.columns and show_meta["start"].notna().any():
        show_meta["start"] = pd.to_datetime(show_meta["start"]).dt.date
    if "end" in show_meta.columns and show_meta["end"].notna().any():
        show_meta["end"] = pd.to_datetime(show_meta["end"]).dt.date
    st.dataframe(show_meta, use_container_width=True, hide_index=True)

    st.subheader("Panel tail")
    st.dataframe(panel_filtered.tail(10).round(3), use_container_width=True)

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
            cluster_df = cluster_df[["cluster", "label", "n_windows",
                                     "growth_z", "inflation_z", "financial_z"]]
            cluster_df[["growth_z", "inflation_z", "financial_z"]] = cluster_df[
                ["growth_z", "inflation_z", "financial_z"]
            ].round(2)
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

# --- Methodology -----------------------------------------------------------

with tab_method:
    st.subheader("Methodology")
    st.markdown(Path(__file__).parent.joinpath("METHODOLOGY.md").read_text())
