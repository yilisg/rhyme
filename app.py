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
from sklearn.cluster import AgglomerativeClustering

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
    _cov_inv,
    _pairwise_mahalanobis,
    _standardize_with_impute,
    block_bootstrap_null_distance,
    cosine_kmeans_similarity,
    embed_2d,
    gmm_similarity,
    primary_similarity,
    regime_probabilities,
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
# chart reads as history, not just a blob of dots.  Authoritative dates:
# OPEC embargo announcement Oct-1973; Volcker fed-funds peak Jun-1981
# (FOMC raised to 20% intra-month, monthly avg peaked at 19.10% in Jun-1981);
# Black Monday Oct 19 1987; S&L peak collapse 1990 (recession start Jul 1990);
# Dot-com NASDAQ peak Mar 10 2000; Lehman bankruptcy Sep 15 2008; COVID-19
# market trough Mar 23 2020; 2022 CPI YoY peak Jun-2022.
CYCLE_EVENTS: list[tuple[str, str]] = [
    ("1973-10-31", "OPEC oil shock"),
    ("1981-06-30", "Volcker peak"),
    ("1987-10-31", "Black Monday"),
    ("1990-07-31", "S&L / 1990 recession"),
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

    # Bayesian regime probabilities (only meaningful for engines that
    # produce centroids in feature space — primary, cosine, gmm).
    regime_probs: np.ndarray | None = None
    if method in ("primary", "cosine", "gmm") and res.cluster_centroids is not None:
        Xs_imp, _, _, valid_mask = _standardize_with_impute(wf.features)
        Xs_imp = Xs_imp[:, valid_mask]
        nan_mask_imp = (~np.isnan(wf.features))[:, valid_mask]
        try:
            vi_full = _cov_inv(Xs_imp, robust=robust)
            regime_probs = regime_probabilities(
                Xs_imp, nan_mask_imp, res.cluster_centroids,
                res.labels, vi_full, res.reference_idx,
            )
        except Exception:
            regime_probs = None

    return {
        "panel": panel,
        "z": z,
        "themes": themes,
        "wf_features": wf.features,
        "wf_end_dates": wf.end_dates,
        "wf_panel_slice": wf.panel_slice,
        "wf_window_size": wf.window_size,
        "wf_columns": wf.columns,
        "wf_column_active_mask": wf.column_active_mask,
        "sim_labels": res.labels,
        "sim_distances": res.distances,
        "sim_reference_idx": res.reference_idx,
        "sim_top_analogs": res.top_analogs,
        "sim_method": res.method,
        "regime_labels": [asdict(r) for r in rlabels],
        "regime_probs": regime_probs,
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


# ---------------------------------------------------------------------------
# Hierarchical similarity, walk-forward labels, block-bootstrap null
# ---------------------------------------------------------------------------


def _hierarchical_distances(
    z_for_features: pd.DataFrame,
    window_size: int,
    n_pca: int,
    robust: bool,
    horizons_periods: dict[str, int],
) -> pd.DataFrame:
    """Compute the today-vs-history Mahalanobis distance per timescale.

    For each horizon (label → window length in periods), we rebuild the
    feature matrix at that window and rank every history end-date by
    distance from the most recent reference.  Returns a DataFrame indexed
    by end-date with one column per horizon: ``dist_<label>`` and
    ``rank_<label>``.
    """
    out_frames: list[pd.DataFrame] = []
    for label, w in horizons_periods.items():
        try:
            wf_h = build_window_features(z_for_features, window_size=w, n_pca=n_pca, robust=robust)
        except ValueError:
            continue
        Xs, _, _, valid = _standardize_with_impute(wf_h.features)
        Xs = Xs[:, valid]
        nan_mask = (~np.isnan(wf_h.features))[:, valid]
        try:
            vi = _cov_inv(Xs, robust=robust)
        except Exception:
            continue
        ref = Xs[-1]
        ref_active = nan_mask[-1]
        d = _pairwise_mahalanobis(Xs, nan_mask, ref, ref_active, vi)
        # Rank ignoring the reference itself + a small gap.
        gap = max(w // 4, 6)
        n = len(d)
        rank = np.full(n, np.nan)
        idx = np.arange(n)
        finite = np.isfinite(d)
        eligible = (np.abs(idx - (n - 1)) >= gap) & finite
        order = np.argsort(np.where(eligible, d, np.inf))
        for r, i in enumerate(order, start=1):
            if not eligible[i]:
                break
            rank[i] = r
        df = pd.DataFrame(
            {f"dist_{label}": d, f"rank_{label}": rank},
            index=wf_h.end_dates,
        )
        out_frames.append(df)
    if not out_frames:
        return pd.DataFrame()
    out = out_frames[0]
    for nxt in out_frames[1:]:
        out = out.join(nxt, how="outer")
    return out


@st.cache_data(show_spinner=False)
def _cached_hierarchical(
    panel_key: str,
    panel_bytes: bytes,
    freq: str,
    zmode: str,
    rolling_years: int,
    min_years: int,
    feature_codes: tuple[str, ...],
    robust: bool,
    transforms_key: str,
    horizons_key: str,  # comma-separated "label:periods" pairs
):
    panel = pd.read_parquet(io.BytesIO(panel_bytes))
    panel.index = pd.to_datetime(panel.index)
    transforms = dict(_decode_pairs(transforms_key))
    horizons = {}
    for token in horizons_key.split(","):
        if ":" in token:
            label, w = token.split(":", 1)
            horizons[label] = int(w)
    z = transform_and_zscore(
        panel, transforms, freq=freq, mode=zmode,
        rolling_years=rolling_years, min_years=min_years, robust=robust,
    )
    z_cols = [f"{c}_z" for c in feature_codes if f"{c}_z" in z.columns]
    return _hierarchical_distances(z[z_cols], window_size=0, n_pca=3, robust=robust, horizons_periods=horizons)


def _walk_forward_labels(
    z_for_features: pd.DataFrame,
    themes: pd.DataFrame,
    individual_z: pd.DataFrame,
    window_size: int,
    n_clusters: int,
    refit_every: int,
    refit_lookback: int,
    label_mode: str,
    robust: bool,
) -> pd.DataFrame:
    """Refit clusters on a rolling window every `refit_every` periods.

    For each rolling fit centered on date T, fit clusters on the most
    recent `refit_lookback` window features ending at or before T; each
    window in the next `refit_every`-period chunk is assigned to the
    closest centroid under that local fit.

    Returns: DataFrame indexed by end-date with columns
    `cluster_local`, `label_local`.
    """
    wf = build_window_features(z_for_features, window_size=window_size, n_pca=3, robust=robust)
    Xs, _, _, valid = _standardize_with_impute(wf.features)
    Xs = Xs[:, valid]

    n = len(wf.end_dates)
    cluster_local = np.full(n, -1, dtype=int)
    label_local = np.empty(n, dtype=object)
    label_local[:] = ""

    # Iterate forward; refit on the trailing `refit_lookback` features
    # whenever we cross a refit boundary.
    last_fit_end = -1
    centroids: np.ndarray | None = None
    member_labels: np.ndarray | None = None
    fit_start_idx: int = 0
    for i in range(n):
        if (i - last_fit_end) >= refit_every or last_fit_end < 0:
            start = max(0, i - refit_lookback + 1)
            if i - start + 1 < max(2 * n_clusters, 24):
                # not enough data to fit yet; defer
                cluster_local[i] = -1
                label_local[i] = ""
                continue
            sub = Xs[start : i + 1]
            try:
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                sub_labels = model.fit_predict(sub)
            except Exception:
                cluster_local[i] = -1
                continue
            centroids = np.vstack(
                [sub.mean(axis=0) if (sub_labels == c).sum() == 0
                 else sub[sub_labels == c].mean(axis=0) for c in range(n_clusters)]
            )
            member_labels = sub_labels
            fit_start_idx = start
            last_fit_end = i
            # Label centroids using cluster-mean theme z over the fit
            # period so the regime tag is the contemporaneous one.
            sub_end_dates = wf.end_dates[start : i + 1]
            local_rlabels = label_clusters(
                themes, sub_labels, sub_end_dates, mode=label_mode,
                robust=robust, individual_z=individual_z,
            )
            local_lm = {rl.cluster: rl.label for rl in local_rlabels}
            # Re-tag this index using closest centroid
        if centroids is None:
            continue
        diffs = Xs[i] - centroids
        # Plain Euclidean on the local fit space — local Mahalanobis is too
        # expensive to recompute every period.
        d2 = np.sum(diffs * diffs, axis=1)
        c = int(np.argmin(d2))
        cluster_local[i] = c
        # Build label map from the most recent fit (memoize via local_lm).
        # Recompute label_clusters when fit changes (it's already cached above).
        # Here we rely on the most recent local_lm in scope — recompute once
        # after each refit by re-evaluating (same as last_fit_end == i path).
        sub_end_dates = wf.end_dates[fit_start_idx : last_fit_end + 1]
        local_rlabels = label_clusters(
            themes, member_labels, sub_end_dates, mode=label_mode,
            robust=robust, individual_z=individual_z,
        )
        local_lm = {rl.cluster: rl.label for rl in local_rlabels}
        label_local[i] = local_lm.get(c, f"#{c}")

    return pd.DataFrame(
        {"cluster_local": cluster_local, "label_local": label_local},
        index=wf.end_dates,
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

DATA_SOURCE_OPTIONS = ["Public", "Private", "Custom"]
data_choice = st.sidebar.selectbox(
    "Panel source",
    DATA_SOURCE_OPTIONS,
    index=0,
    help=(
        "Public: built-in FRED + Yahoo panel.  "
        "Private: read tabula's long-format parquet (defaults to its "
        "standard output path).  "
        "Custom: upload a CSV or parquet (wide or long format)."
    ),
)

uploaded = None
private_path: str | None = None
TABULA_DEFAULT_PATH = "/Users/yili/Desktop/Claude/tabula/data/output/tabula_panel.parquet"
if data_choice == "Private":
    private_path = st.sidebar.text_input(
        "Tabula parquet path",
        value=TABULA_DEFAULT_PATH,
        help="Long-format parquet: series_id, observation_date, value, source.",
    )
    private_upload = st.sidebar.file_uploader(
        "...or upload a tabula parquet directly",
        type=["parquet"],
        key="private_upload",
    )
    if private_upload is not None:
        uploaded = private_upload
elif data_choice == "Custom":
    uploaded = st.sidebar.file_uploader(
        "File (CSV: first column = date; parquet: long-format with series_id/observation_date/value)",
        type=["csv", "parquet", "json"],
    )

long_term_model = st.sidebar.checkbox(
    "Long-term model (allow pre-2000 history)",
    value=False,
    help=(
        "Off: panel observations before 2000-01-01 are filtered out — this "
        "matches the public FRED+Yahoo coverage floor for most series.  "
        "On: lets full GFD-sourced (or other) deep history through.  Useful "
        "when you've loaded a private long panel that goes back to e.g. 1950."
    ),
)

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

with st.sidebar.expander("Advanced (analog mechanics)", expanded=False):
    walk_forward_labels_on = st.checkbox(
        "Walk-forward regime labels",
        value=False,
        help=(
            "Off: clusters are fit on the full panel up to today (one snapshot, "
            "label space anchored to current data).  "
            "On: clusters are refit every 12 periods on a rolling 360-period "
            "(or full available) trailing window; each window's label is the "
            "centroid it's closest to under the contemporaneous fit.  Slower "
            "but reflects how the regime would have been classified at the time."
        ),
    )
    hierarchical_on = st.checkbox(
        "Hierarchical similarity (3y / 10y / 30y)",
        value=False,
        help=(
            "Compute today-vs-history Mahalanobis distance separately for "
            "short / medium / long lookback windows.  Surfaces in the Analogs "
            "tab — today's regime can rhyme with different things on "
            "different timescales."
        ),
    )
    bootstrap_on = st.checkbox(
        "Block-bootstrap significance (slow)",
        value=False,
        help=(
            "Permute the panel in 24-month blocks 1000× and report the "
            "distribution of best-analog distance under random data.  Flags "
            "if today's best-analog distance is *not* in the tail of the "
            "null — i.e. we'd see something this close even if the data were "
            "random."
        ),
    )
    comparison_on = st.checkbox(
        "Comparison view (pick two dates)",
        value=False,
        help=(
            "Pin two reference dates and overlay their themes / clocks / "
            "cycle plots side-by-side."
        ),
    )

compare_dates: tuple[pd.Timestamp, pd.Timestamp] | None = None
if comparison_on:
    with st.sidebar.expander("Comparison dates", expanded=True):
        d1_str = st.text_input("Date A (YYYY-MM-DD)", value="2008-09-30")
        d2_str = st.text_input("Date B (YYYY-MM-DD)", value="2022-06-30")
        try:
            compare_dates = (pd.Timestamp(d1_str), pd.Timestamp(d2_str))
        except Exception:
            compare_dates = None
            st.sidebar.warning("Bad date — use YYYY-MM-DD.")

# ---------------------------------------------------------------------------
# Build panel
# ---------------------------------------------------------------------------


def _reshape_long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format panel (series_id / observation_date / value) into
    rhyme's wide schema (date index, one column per series).

    Tabula-style long format is detected by the presence of all three of
    series_id, observation_date, value.  Other column names (case-
    insensitive aliases) are also accepted: code/series, date, val.
    """
    cols = {c.lower(): c for c in long_df.columns}
    date_col = (
        cols.get("observation_date")
        or cols.get("date")
        or cols.get("obs_date")
        or cols.get("period")
    )
    series_col = (
        cols.get("series_id")
        or cols.get("series")
        or cols.get("code")
        or cols.get("ticker")
    )
    value_col = cols.get("value") or cols.get("val") or cols.get("v")
    if not (date_col and series_col and value_col):
        raise ValueError(
            "long-format requires columns for date, series_id, value (got "
            f"{list(long_df.columns)})"
        )
    work = long_df[[date_col, series_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    wide = (
        work.pivot_table(index=date_col, columns=series_col, values=value_col, aggfunc="last")
        .sort_index()
    )
    wide.columns.name = None
    return wide


def _load_upload(file) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Read user-supplied panel from CSV / JSON / parquet.  Auto-detects
    long-format parquet (tabula's schema) and pivots into wide."""
    name = getattr(file, "name", "").lower()
    if name.endswith(".parquet"):
        raw = pd.read_parquet(file)
        if "value" in {c.lower() for c in raw.columns}:
            wide = _reshape_long_to_wide(raw)
        else:
            # already wide; date is the index or the first column
            if not isinstance(raw.index, pd.DatetimeIndex):
                date_col = raw.columns[0]
                raw[date_col] = pd.to_datetime(raw[date_col])
                raw = raw.set_index(date_col)
            wide = raw.sort_index().apply(pd.to_numeric, errors="coerce")
    elif name.endswith(".json"):
        raw = pd.read_json(file)
        date_col = raw.columns[0]
        raw[date_col] = pd.to_datetime(raw[date_col])
        wide = raw.set_index(date_col).sort_index().apply(pd.to_numeric, errors="coerce")
    else:
        raw = pd.read_csv(file)
        if "value" in {c.lower() for c in raw.columns} and (
            "series_id" in {c.lower() for c in raw.columns}
            or "series" in {c.lower() for c in raw.columns}
        ):
            wide = _reshape_long_to_wide(raw)
        else:
            date_col = raw.columns[0]
            raw[date_col] = pd.to_datetime(raw[date_col])
            wide = raw.set_index(date_col).sort_index().apply(pd.to_numeric, errors="coerce")
    wide = wide.dropna(how="all")
    transforms = infer_transforms(wide)
    buckets = {c: "monetary" for c in wide.columns}
    return wide, transforms, buckets


def _load_private_path(path: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Read tabula-style long-format parquet from a filesystem path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Private parquet not found: {path}")
    raw = pd.read_parquet(p)
    if "value" in {c.lower() for c in raw.columns}:
        wide = _reshape_long_to_wide(raw)
    else:
        if not isinstance(raw.index, pd.DatetimeIndex):
            date_col = raw.columns[0]
            raw[date_col] = pd.to_datetime(raw[date_col])
            raw = raw.set_index(date_col)
        wide = raw.sort_index().apply(pd.to_numeric, errors="coerce")
    wide = wide.dropna(how="all")
    transforms = infer_transforms(wide)
    buckets = {c: "monetary" for c in wide.columns}
    return wide, transforms, buckets


if data_choice == "Public":
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
    panel_source_name = "public"
    # The old stale/short-series filter (a workaround for dropna(how="any")
    # in build_window_features) is gone now that features are NaN-tolerant.
    # A series with limited coverage simply contributes to whatever windows
    # it overlaps with — see rhyme_lib/features.py for the active-mask
    # design.  We still drop columns that have *no* data anywhere.
    feature_codes = [c for c in feature_codes if panel[c].notna().any()]
elif data_choice in ("Private", "Custom"):
    src_label = data_choice.lower()
    try:
        if uploaded is not None:
            panel, transforms, buckets = _load_upload(uploaded)
            panel_source_name = getattr(uploaded, "name", f"{src_label}-upload")
        elif data_choice == "Private" and private_path:
            panel, transforms, buckets = _load_private_path(private_path)
            panel_source_name = f"private:{Path(private_path).name}"
        else:
            st.info(
                "Choose a file or set the path in the sidebar.  "
                "Private mode reads tabula's long-format parquet "
                "(series_id / observation_date / value)."
            )
            st.stop()
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Failed to load {src_label} panel: {e}")
        st.stop()
    meta = pd.DataFrame(
        [
            {
                "code": c,
                "bucket": buckets[c],
                "transform": transforms[c],
                "source": src_label,
                "n_obs": int(panel[c].notna().sum()),
                "start": panel[c].dropna().index.min() if panel[c].notna().any() else None,
                "end":   panel[c].dropna().index.max() if panel[c].notna().any() else None,
            }
            for c in panel.columns
        ]
    )
    all_codes = list(panel.columns)
    feature_codes = [c for c in panel.columns if panel[c].notna().any()]
else:
    st.error(f"Unknown panel source: {data_choice}")
    st.stop()

# Long-term-model toggle: enforce a 2000-01-01 floor when off so we mimic
# the public-source coverage floor regardless of the underlying source.
if not long_term_model:
    floor = pd.Timestamp("2000-01-01")
    if panel.index.min() < floor:
        panel = panel.loc[panel.index >= floor]
        if "start" in meta.columns:
            meta = meta.assign(
                start=meta["start"].apply(
                    lambda d: max(pd.Timestamp(d), floor) if pd.notna(d) else d
                )
            )

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
# Optional walk-forward labels + hierarchical timescale distances
# ---------------------------------------------------------------------------

walk_forward_label_df: pd.DataFrame | None = None
if walk_forward_labels_on:
    try:
        # Use the same z-panel that fed the main feature build.
        z_local = result["z"]
        themes_local = result["themes"]
        z_cols = [f"{c}_z" for c in feature_codes if f"{c}_z" in z_local.columns]
        # Refit every 12 periods on a 360-period (or available) trailing window.
        refit_every = 12
        refit_lookback = 360
        walk_forward_label_df = _walk_forward_labels(
            z_for_features=z_local[z_cols],
            themes=themes_local,
            individual_z=z_local,
            window_size=window_size,
            n_clusters=n_clusters,
            refit_every=refit_every,
            refit_lookback=refit_lookback,
            label_mode=("market" if mode == "Market" else "macro"),
            robust=robust,
        )
    except Exception as e:
        st.sidebar.warning(f"Walk-forward labels disabled: {e}")
        walk_forward_label_df = None

hierarchical_df: pd.DataFrame | None = None
if hierarchical_on:
    # Periods per horizon depend on freq.  Approximate: 1y = 12 (M) / 52 (W).
    if freq == "M":
        h_periods = {"3y": 36, "10y": 120, "30y": 360}
    elif freq == "W":
        h_periods = {"3y": 156, "10y": 520, "30y": 1560}
    else:  # daily
        h_periods = {"3y": 756, "10y": 2520, "30y": 7560}
    horizons_key = ",".join(f"{k}:{v}" for k, v in h_periods.items())
    try:
        hierarchical_df = _cached_hierarchical(
            panel_key=panel_source_name,
            panel_bytes=_panel_bytes.getvalue(),
            freq=freq,
            zmode=zmode,
            rolling_years=rolling_years,
            min_years=min_years,
            feature_codes=tuple(feature_codes),
            robust=robust,
            transforms_key=_encode_pairs(
                {k: v for k, v in transforms.items() if k in all_codes}
            ),
            horizons_key=horizons_key,
        )
    except Exception as e:
        st.sidebar.warning(f"Hierarchical similarity disabled: {e}")
        hierarchical_df = None

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
    # If walk-forward labels are on, show the contemporaneous label.
    if walk_forward_label_df is not None and ref_end in walk_forward_label_df.index:
        wf_label = walk_forward_label_df.at[ref_end, "label_local"]
        col2.metric("Today's regime (walk-forward)", wf_label or ref_label)
    else:
        col2.metric("Today's regime", ref_label)
    col3.metric("Windows analyzed", f"{len(result['wf_end_dates']):,}")
    col4.metric("Method", METHOD_LABELS[method].split(":", 1)[-1].strip())

    # Bayesian regime probabilities — softmax over Mahalanobis distance to
    # cluster centroids.
    regime_probs = result.get("regime_probs")
    if regime_probs is not None and len(regime_probs) == len(regime_labels):
        st.subheader("Regime probabilities")
        st.caption(
            "Softmax over Mahalanobis distance to each cluster centroid: "
            "p_k ∝ exp(-d_k² / τ).  τ = median within-cluster squared distance."
        )
        prob_df = pd.DataFrame({
            "regime": [rl["label"] for rl in regime_labels],
            "prob": regime_probs,
        }).sort_values("prob", ascending=True)
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=prob_df["prob"], y=prob_df["regime"], orientation="h",
            marker=dict(color="#FFD700", line=dict(color="#D10000", width=1)),
            text=[f"{p:.1%}" for p in prob_df["prob"]],
            textposition="outside",
        ))
        fig_prob.update_layout(
            height=max(220, 50 * len(prob_df) + 80),
            xaxis=dict(title="probability", range=[0, 1.05], tickformat=".0%"),
            yaxis_title=None,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("Top 5 analogs")
    compact = top_analogs.head(5).copy()
    compact["window_start"] = compact["end_date"] - pd.Timedelta(weeks=window_size if freq == "W" else window_size * 4)
    display = compact[["rank", "window_start", "end_date", "distance", "label"]].copy()
    display["window_start"] = display["window_start"].dt.strftime("%Y-%m-%d")
    display["end_date"] = display["end_date"].dt.strftime("%Y-%m-%d")
    display["distance"] = display["distance"].apply(lambda v: f"{v:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Why this regime (top contributing z-scores) -----------------------
    st.subheader("Why this regime")
    st.caption(
        "The series whose z-score at the reference window is most extreme — "
        "directly falsifiable: if these readings change, the regime flag "
        "should change."
    )
    z_today = result["z"].reindex(result["wf_end_dates"]).iloc[-1]
    z_today = z_today.dropna()
    if not z_today.empty:
        z_sorted = z_today.sort_values()
        top_neg = z_sorted.head(3)
        top_pos = z_sorted.tail(3).iloc[::-1]
        cwhy1, cwhy2 = st.columns(2)
        with cwhy1:
            st.markdown("**Most positive (overheated / above-trend)**")
            why_pos = pd.DataFrame({
                "series": [c[:-2] for c in top_pos.index],
                "z": top_pos.values.round(2),
            })
            st.dataframe(why_pos, use_container_width=True, hide_index=True)
        with cwhy2:
            st.markdown("**Most negative (cooling / below-trend)**")
            why_neg = pd.DataFrame({
                "series": [c[:-2] for c in top_neg.index],
                "z": top_neg.values.round(2),
            })
            st.dataframe(why_neg, use_container_width=True, hide_index=True)

    st.subheader("Theme z-scores (last 4 observations)")
    st.dataframe(result["themes"].dropna(how="all").tail(4).round(2), use_container_width=True)

    # --- Block-bootstrap null distribution --------------------------------
    if bootstrap_on:
        st.subheader("Block-bootstrap significance test")
        st.caption(
            "Permute the panel in 24-month blocks 1000× and record the "
            "best-analog Mahalanobis distance under each permutation.  "
            "If today's best-analog distance is at the small-tail end of "
            "this null, the analog is meaningfully better than chance; if "
            "it's near the median, today doesn't rhyme with anything in "
            "particular more than random data would."
        )
        try:
            Xs_b, _, _, valid_b = _standardize_with_impute(result["wf_features"])
            Xs_b = Xs_b[:, valid_b]
            nan_mask_b = (~np.isnan(result["wf_features"]))[:, valid_b]
            vi_b = _cov_inv(Xs_b, robust=robust)
            block = 24 if freq == "M" else (24 * 4 if freq == "W" else 24 * 21)
            null_dists = block_bootstrap_null_distance(
                Xs_b, nan_mask_b, vi_b, ref_idx=ref_idx,
                block_size=block, n_reps=1000, rng_seed=0,
                min_gap=max(window_size // 4, 6),
            )
            null_dists = null_dists[np.isfinite(null_dists)]
            actual = float(top_analogs["distance"].iloc[0])
            pct = float((null_dists <= actual).mean()) if len(null_dists) else float("nan")

            fig_null = go.Figure()
            fig_null.add_trace(go.Histogram(
                x=null_dists, nbinsx=40,
                marker=dict(color="rgba(150,150,150,0.6)", line=dict(color="#888", width=1)),
                name="null (block-bootstrap)",
            ))
            fig_null.add_vline(
                x=actual, line_color="#D10000", line_width=2,
                annotation_text=f"today's best = {actual:.3f}",
                annotation_position="top",
            )
            fig_null.update_layout(
                height=320,
                xaxis_title="best-analog Mahalanobis distance",
                yaxis_title="count under null",
                showlegend=False,
            )
            st.plotly_chart(fig_null, use_container_width=True)
            verdict = (
                "in the **small-tail** of the null — the best analog is meaningfully closer than random data"
                if pct < 0.05
                else "**not in the tail** — today doesn't rhyme more than random data would"
                if pct > 0.5
                else "in the lower-half of the null — modestly informative"
            )
            st.caption(
                f"Today's best-analog distance is at the **{pct:.1%} percentile** "
                f"of the 24-{('month' if freq == 'M' else 'period')} block-bootstrap null "
                f"({len(null_dists)} reps).  Verdict: {verdict}."
            )
        except Exception as e:
            st.warning(f"Bootstrap failed: {e}")

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

        # Comparison-view overlay: pin two dates with crimson + teal markers.
        if compare_dates is not None:
            for d, color, name in [
                (compare_dates[0], "#D10000", f"A: {compare_dates[0].date()}"),
                (compare_dates[1], "#0080A0", f"B: {compare_dates[1].date()}"),
            ]:
                if d not in cycle_df.index:
                    pos_idx = cycle_df.index.get_indexer([d], method="nearest")
                    if len(pos_idx) and pos_idx[0] >= 0:
                        d_use = cycle_df.index[pos_idx[0]]
                    else:
                        continue
                else:
                    d_use = d
                row = cycle_df.loc[d_use]
                fig.add_trace(go.Scatter(
                    x=[row[x_col]], y=[row[y_col]],
                    mode="markers+text",
                    marker=dict(size=18, color=color, symbol="circle-open",
                                line=dict(color=color, width=3)),
                    text=[f"  {name}"], textposition="middle right",
                    textfont=dict(size=11, color=color),
                    name=name, showlegend=False,
                ))

        st.plotly_chart(fig, use_container_width=True)

        if not trail.empty:
            cap = trail.iloc[-1]
            st.caption(
                f"**Today ({cap['date'].date()}):** {x_title} = {cap['x']:+.2f}, "
                f"{y_title} = {cap['y']:+.2f}.  {note}  "
                f"Grey dots = every historical period. Diamonds = landmark events. "
                f"Gold trail = T-12M → T-6M → T-3M → T-1M → today."
            )

        # Comparison view: show themes side-by-side around each pinned date.
        if compare_dates is not None:
            st.markdown("---")
            st.subheader("Comparison: themes & individual z at the pinned dates")
            themes_df = result["themes"].dropna(how="all")
            z_df = result["z"]
            comp_rows = []
            for d in compare_dates:
                if d in themes_df.index:
                    t_row = themes_df.loc[d]
                else:
                    pos = themes_df.index.get_indexer([d], method="nearest")
                    if len(pos) and pos[0] >= 0:
                        t_row = themes_df.iloc[pos[0]]
                    else:
                        continue
                rec = {"date": pd.Timestamp(t_row.name).date()}
                for col in themes_df.columns:
                    rec[col] = round(float(t_row[col]), 2) if pd.notna(t_row[col]) else ""
                comp_rows.append(rec)
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

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

    # --- Hierarchical similarity table -------------------------------------
    if hierarchical_df is not None and not hierarchical_df.empty:
        st.subheader("Hierarchical similarity (3y / 10y / 30y)")
        st.caption(
            "Today's regime can rhyme with different histories on different "
            "timescales.  Each column is a Mahalanobis distance computed on "
            "features built from the labelled window length, plus today's "
            "rank against history at that timescale."
        )
        # Build a reverse-distance ranked table — the K nearest end-dates at
        # each horizon, joined on date.
        h_labels = [c.replace("dist_", "") for c in hierarchical_df.columns if c.startswith("dist_")]
        rows: list[dict] = []
        for h in h_labels:
            d_col = f"dist_{h}"
            r_col = f"rank_{h}"
            sub = hierarchical_df[[d_col, r_col]].dropna()
            if sub.empty:
                continue
            top5 = sub.nsmallest(5, d_col).reset_index().rename(columns={"index": "end_date"})
            for r in top5.itertuples():
                rows.append({
                    "horizon": h,
                    "rank": int(getattr(r, r_col)),
                    "end_date": getattr(r, "end_date").strftime("%Y-%m-%d"),
                    "distance": f"{getattr(r, d_col):.3f}",
                })
        if rows:
            st.dataframe(
                pd.DataFrame(rows).sort_values(["horizon", "rank"]),
                use_container_width=True,
                hide_index=True,
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
