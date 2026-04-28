"""Build window-level feature vectors from the z-scored series panel.

Each window → feature vector = concat(
    per-series moments [mean, std, skew, lag1-autocorr, start-end drift],
    upper-triangle of cross-series correlation matrix,
    first k PCA factor scores of the raw (within-window) values,
)

NaN-tolerant design (2026-04 rewrite).  The previous implementation called
``zpanel.dropna(how="any")`` and collapsed the panel to its shortest
NaN-free intersection.  That produced ~42 monthly windows in the default
panel.  The current implementation:

* Aligns on the union of dates where *at least one* feature column is
  observed.
* For each window, computes per-column moments only on columns active in
  that window (≥ ``min_obs_in_window`` non-NaN observations).
* Builds a fixed-length feature vector keyed by the global column list,
  filling with ``NaN`` whenever a column is inactive in a particular
  window.
* Records a per-window ``active_mask`` so downstream similarity engines
  can compute distances on the intersection of features active in the
  pair / reference set.

The panel feed-in is therefore a *raw* z-panel (no upstream dropna), and
``WindowFeatures.features`` may contain NaNs by design.  Skip windows
with fewer than ``min_active_features`` columns active (degenerate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class WindowFeatures:
    end_dates: pd.DatetimeIndex
    features: np.ndarray              # (n_windows, n_features) — may contain NaNs
    feature_names: list[str]
    panel_slice: pd.DataFrame         # the z-scored panel fed in (full, may have NaN)
    window_size: int
    column_active_mask: np.ndarray    # (n_windows, n_columns) — True if column has data in that window
    columns: list[str]                # column names corresponding to column_active_mask
    feature_to_column: np.ndarray     # (n_features,) → column index, -1 for cross/PCA features


def _moments(block: np.ndarray, robust: bool = False) -> np.ndarray:
    """block: (T, d) z-scored values for one window. Returns (5d,).

    NaN-aware: each column's moments are computed only on its non-NaN
    observations.  Columns with < 3 valid obs return NaN moments.
    If `robust`, uses median and MAD (×1.4826) instead of mean and std."""
    T, d = block.shape
    valid_count = (~np.isnan(block)).sum(axis=0)

    with np.errstate(all="ignore"):
        if robust:
            means = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - means), axis=0) * 1.4826
            nan_std = np.nanstd(block, axis=0)
            stds = np.where(np.isfinite(mad) & (mad > 0), mad, nan_std)
        else:
            means = np.nanmean(block, axis=0)
            stds = np.nanstd(block, axis=0)

    skews = np.array([_safe_skew(block[:, j]) for j in range(d)])
    ac1 = np.array([_safe_ac1(block[:, j]) for j in range(d)])

    drift = np.full(d, np.nan)
    for j in range(d):
        col = block[:, j]
        valid = ~np.isnan(col)
        if valid.sum() >= 2:
            idx = np.where(valid)[0]
            drift[j] = float(col[idx[-1]] - col[idx[0]])

    # Mask out columns with too few obs (any moment is NaN by construction).
    too_few = valid_count < 3
    means = np.where(too_few, np.nan, means)
    stds = np.where(too_few, np.nan, stds)
    skews = np.where(too_few, np.nan, skews)
    ac1 = np.where(too_few, np.nan, ac1)
    drift = np.where(too_few, np.nan, drift)
    return np.concatenate([means, stds, skews, ac1, drift])


def _safe_skew(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    m = x.mean()
    s = x.std()
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _safe_ac1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    a = x[:-1]
    b = x[1:]
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0 or not np.isfinite(sa) or not np.isfinite(sb):
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _corr_upper(
    block: np.ndarray,
    robust: bool = False,
    min_pair_obs: int = 12,
) -> np.ndarray:
    """Upper triangle of the within-window correlation matrix, NaN-aware.

    For each pair (i, j) the correlation is computed on rows where both
    i and j are non-NaN.  Returns NaN when fewer than ``min_pair_obs``
    common observations exist; downstream consumers can either drop
    those entries or treat them as zero.
    """
    T, d = block.shape
    if d < 2:
        return np.array([])
    iu_i, iu_j = np.triu_indices(d, k=1)
    out = np.empty(len(iu_i))
    for k, (i, j) in enumerate(zip(iu_i, iu_j)):
        a = block[:, i]
        b = block[:, j]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < min_pair_obs:
            out[k] = np.nan
            continue
        a = a[m]
        b = b[m]
        if robust:
            ar = np.argsort(np.argsort(a)).astype(float)
            br = np.argsort(np.argsort(b)).astype(float)
            sa = ar.std()
            sb = br.std()
            if sa == 0 or sb == 0:
                out[k] = 0.0
                continue
            out[k] = float(((ar - ar.mean()) * (br - br.mean())).mean() / (sa * sb))
        else:
            sa = a.std()
            sb = b.std()
            if sa == 0 or sb == 0:
                out[k] = 0.0
                continue
            out[k] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return out


def _pca_scores_nanaware(
    block: np.ndarray,
    n_pca: int,
) -> np.ndarray:
    """First ``n_pca`` PCA factor scores at the window's end, NaN-tolerant.

    PCA is fit only on columns active across the entire window.  Columns
    with any NaN inside the window are dropped from the PCA basis (their
    information is already captured in the per-series moments).  Returns
    an n_pca-vector of NaN if too few active columns to fit.
    """
    valid_cols = ~np.isnan(block).any(axis=0)
    sub = block[:, valid_cols]
    n_obs, n_feats = sub.shape
    n_components = min(n_pca, max(0, n_feats - 1), max(0, n_obs - 1))
    if n_components < 1:
        return np.full(n_pca, np.nan)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(sub)[-1]
    if len(scores) < n_pca:
        scores = np.concatenate([scores, np.zeros(n_pca - len(scores))])
    return scores


def build_window_features(
    zpanel: pd.DataFrame,
    window_size: int,
    n_pca: int = 3,
    robust: bool = False,
    min_active_features: int = 5,
    min_obs_in_window: int | None = None,
) -> WindowFeatures:
    """Build a NaN-tolerant feature matrix.

    Args:
        zpanel: wide z-score panel (columns are <code>_z, index is date).
        window_size: rolling window length in periods.
        n_pca: number of PCA factor scores to retain.
        robust: use median/MAD/Spearman instead of mean/std/Pearson.
        min_active_features: a window is dropped if fewer than this many
            columns are active.  Default 5.
        min_obs_in_window: a column counts as "active" in a window if it
            has at least this many non-NaN observations.  Default
            ``max(window_size // 2, 6)``.

    Returns: ``WindowFeatures`` with possibly-NaN ``features``.  Downstream
    similarity engines must handle NaNs by intersecting the active feature
    sets pairwise (see ``rhyme_lib.similarity``).
    """
    panel = zpanel.sort_index()
    if min_obs_in_window is None:
        min_obs_in_window = max(window_size // 2, 6)

    # Use the union of dates where any column is observed.  We do NOT
    # dropna(how="any") — that's the structural change.
    all_obs_mask = panel.notna().any(axis=1)
    panel = panel.loc[all_obs_mask]
    if len(panel) < window_size + 10:
        raise ValueError(
            f"not enough rows ({len(panel)}) for window_size={window_size}"
        )

    columns = panel.columns.tolist()
    d = len(columns)
    values = panel.to_numpy(dtype=float)
    n_windows = len(panel) - window_size + 1

    moment_names = (
        [f"mean_{c}" for c in columns]
        + [f"std_{c}" for c in columns]
        + [f"skew_{c}" for c in columns]
        + [f"ac1_{c}" for c in columns]
        + [f"drift_{c}" for c in columns]
    )
    iu_i, iu_j = np.triu_indices(d, k=1)
    corr_names = [f"corr_{columns[i]}__{columns[j]}" for i, j in zip(iu_i, iu_j)]
    pca_names = [f"pc{i + 1}" for i in range(n_pca)]
    feature_names = moment_names + corr_names + pca_names

    # Map each feature index to a column index (or -1 for cross/PCA).
    feat_to_col = (
        list(range(d)) * 5  # 5 moments × d columns
        + [-1] * len(corr_names)
        + [-1] * n_pca
    )
    feat_to_col = np.asarray(feat_to_col, dtype=int)

    end_dates_keep: list[pd.Timestamp] = []
    feat_rows: list[np.ndarray] = []
    active_rows: list[np.ndarray] = []

    for i in range(n_windows):
        block = values[i : i + window_size]
        # Per-column active mask = enough valid obs in this window.
        col_valid = (~np.isnan(block)).sum(axis=0) >= min_obs_in_window
        n_active = int(col_valid.sum())
        if n_active < min_active_features:
            continue

        # Build feature vector with NaN for inactive moment columns; cross
        # entries between inactive columns become NaN automatically via
        # ``_corr_upper``.
        m = _moments(block, robust=robust)
        c = _corr_upper(block, robust=robust, min_pair_obs=min_obs_in_window // 2)
        scores = _pca_scores_nanaware(block, n_pca)
        feat = np.concatenate([m, c, scores])

        # Force any moment of an inactive column to NaN (defensive).
        # m is laid out as [means(d), stds(d), skews(d), ac1(d), drift(d)].
        for k in range(5):
            seg = feat[k * d : (k + 1) * d]
            seg[~col_valid] = np.nan
            feat[k * d : (k + 1) * d] = seg

        feat_rows.append(feat)
        active_rows.append(col_valid)
        end_dates_keep.append(panel.index[i + window_size - 1])

    if not feat_rows:
        raise ValueError(
            f"no windows had >= {min_active_features} active features; "
            f"check panel coverage / window_size."
        )

    features = np.vstack(feat_rows)
    column_active_mask = np.vstack(active_rows)

    return WindowFeatures(
        end_dates=pd.DatetimeIndex(end_dates_keep),
        features=features,
        feature_names=feature_names,
        panel_slice=panel,
        window_size=window_size,
        column_active_mask=column_active_mask,
        columns=columns,
        feature_to_column=feat_to_col,
    )
