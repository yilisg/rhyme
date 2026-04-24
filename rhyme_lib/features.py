"""Build window-level feature vectors from the z-scored series panel.

Each window → feature vector = concat(
    per-series moments [mean, std, skew, lag1-autocorr, start-end drift],
    upper-triangle of cross-series correlation matrix,
    first k PCA factor scores of the raw (within-window) values,
)

The pipeline returns the feature matrix, the window end-dates, and the raw
within-window slices keyed by end-date (useful for plotting/debug).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class WindowFeatures:
    end_dates: pd.DatetimeIndex
    features: np.ndarray
    feature_names: list[str]
    panel_slice: pd.DataFrame  # the z-scored panel fed in (aligned, NaN-free)
    window_size: int


def _moments(block: np.ndarray) -> np.ndarray:
    """block: (T, d) z-scored values for one window. Returns (5d,)."""
    T, d = block.shape
    means = block.mean(axis=0)
    stds = block.std(axis=0)
    skews = np.array([_safe_skew(block[:, j]) for j in range(d)])
    ac1 = np.array([_safe_ac1(block[:, j]) for j in range(d)])
    drift = block[-1] - block[0]
    return np.concatenate([means, stds, skews, ac1, drift])


def _safe_skew(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _safe_ac1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return 0.0
    a = x[:-1]
    b = x[1:]
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0 or not np.isfinite(sa) or not np.isfinite(sb):
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _corr_upper(block: np.ndarray) -> np.ndarray:
    d = block.shape[1]
    if d < 2:
        return np.array([])
    c = np.corrcoef(block.T)
    iu = np.triu_indices(d, k=1)
    return c[iu]


def build_window_features(
    zpanel: pd.DataFrame,
    window_size: int,
    n_pca: int = 3,
) -> WindowFeatures:
    clean = zpanel.dropna(how="any").sort_index()
    if len(clean) < window_size + 10:
        raise ValueError(
            f"not enough NaN-free rows ({len(clean)}) for window_size={window_size}"
        )

    series = clean.columns.tolist()
    d = len(series)
    values = clean.to_numpy(dtype=float)
    n_windows = len(clean) - window_size + 1

    moment_names = (
        [f"mean_{c}" for c in series]
        + [f"std_{c}" for c in series]
        + [f"skew_{c}" for c in series]
        + [f"ac1_{c}" for c in series]
        + [f"drift_{c}" for c in series]
    )
    corr_names = [
        f"corr_{series[i]}__{series[j]}"
        for i in range(d)
        for j in range(i + 1, d)
    ]
    pca_names = [f"pc{i + 1}" for i in range(n_pca)]
    feature_names = moment_names + corr_names + pca_names

    feat_list = []
    for i in range(n_windows):
        block = values[i : i + window_size]
        m = _moments(block)
        c = _corr_upper(block)
        pca = PCA(n_components=min(n_pca, d, window_size - 1))
        scores = pca.fit_transform(block)[-1]
        if len(scores) < n_pca:
            scores = np.concatenate([scores, np.zeros(n_pca - len(scores))])
        feat_list.append(np.concatenate([m, c, scores]))

    features = np.vstack(feat_list)
    bad = ~np.isfinite(features)
    if bad.any():
        features[bad] = 0.0

    return WindowFeatures(
        end_dates=clean.index[window_size - 1 :],
        features=features,
        feature_names=feature_names,
        panel_slice=clean,
        window_size=window_size,
    )
