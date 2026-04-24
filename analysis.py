"""Core analytics for Rhyme: build rolling windows over a multi-series return
panel, cluster them, and find the historical windows most similar to the
reference (most-recent) window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowSet:
    end_dates: pd.DatetimeIndex
    features: np.ndarray
    scaler: StandardScaler


def load_panel(file) -> pd.DataFrame:
    name = getattr(file, "name", "").lower()
    if name.endswith(".json"):
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def build_windows(panel: pd.DataFrame, window_size: int) -> WindowSet:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if len(panel) < window_size:
        raise ValueError(
            f"Not enough rows ({len(panel)}) for window_size={window_size}"
        )

    values = panel.ffill().bfill().to_numpy(dtype=float)
    n_rows, n_series = values.shape
    n_windows = n_rows - window_size + 1

    raw = np.empty((n_windows, window_size * n_series), dtype=float)
    for i in range(n_windows):
        block = values[i : i + window_size]
        block = (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-9)
        raw[i] = block.reshape(-1)

    scaler = StandardScaler()
    features = scaler.fit_transform(raw)
    end_dates = panel.index[window_size - 1 :]
    return WindowSet(end_dates=end_dates, features=features, scaler=scaler)


def cluster_windows(ws: WindowSet, k: int, random_state: int = 0) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return model.fit_predict(ws.features)


def find_analogs(
    ws: WindowSet,
    reference_idx: int,
    top_k: int = 10,
    min_gap: int = None,
) -> pd.DataFrame:
    """Return the top_k historical windows most similar to the reference.

    min_gap prevents overlapping windows from dominating — default is window
    size itself, inferred from feature dimensionality."""
    ref = ws.features[reference_idx]
    dists = np.linalg.norm(ws.features - ref, axis=1)

    if min_gap is None:
        min_gap = 1
    candidate_idx = np.arange(len(ws.end_dates))
    mask = np.abs(candidate_idx - reference_idx) >= min_gap
    ranked = candidate_idx[mask][np.argsort(dists[mask])]

    top = ranked[:top_k]
    return pd.DataFrame(
        {
            "end_date": ws.end_dates[top],
            "distance": dists[top],
        }
    )


def project_2d(ws: WindowSet, random_state: int = 0) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(ws.features)
