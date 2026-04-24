"""Similarity and clustering engine.

Primary methodology: Mahalanobis distance (Ledoit-Wolf shrunk covariance)
on the window feature matrix, plus Ward hierarchical clustering. This is
the Kritzman-Page-Turkington "turbulence" formulation extended to a richer
feature set.

Secondary methodology: Shape-based distance (k-Shape / SBD) on the raw
z-scored panel slice, plus HDBSCAN density-based clustering.

Embeddings for the scatter view: UMAP (primary), t-SNE, PCA (toggles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

from .features import WindowFeatures

Method = Literal["primary", "secondary", "cosine", "gmm"]


@dataclass
class SimilarityResult:
    labels: np.ndarray              # cluster label per window
    distances: np.ndarray           # distance from each window to reference
    reference_idx: int
    top_analogs: pd.DataFrame       # end_date, distance, cluster, rank
    method: Method


def _cov_inv(X: np.ndarray, robust: bool = False) -> np.ndarray:
    if robust:
        # MinCovDet needs n > 2*d; fall back to Ledoit-Wolf if not enough data.
        n, d = X.shape
        if n > 2 * d + 10:
            try:
                cov = MinCovDet(support_fraction=None, random_state=0).fit(X).covariance_
            except Exception:
                cov = LedoitWolf().fit(X).covariance_
        else:
            cov = LedoitWolf().fit(X).covariance_
    else:
        cov = LedoitWolf().fit(X).covariance_
    d = cov.shape[0]
    reg = 1e-6 * np.trace(cov) / max(d, 1)
    return np.linalg.pinv(cov + reg * np.eye(d))


def _standardize_features(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs


def _build_top_df(
    dists: np.ndarray,
    labels: np.ndarray,
    end_dates: pd.DatetimeIndex,
    ref_idx: int,
    gap: int,
    top_k: int,
) -> pd.DataFrame:
    idx = np.arange(len(dists))
    mask = np.abs(idx - ref_idx) >= gap
    ranked = idx[mask][np.argsort(dists[mask])]
    top = ranked[:top_k]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(top) + 1),
            "end_date": end_dates[top],
            "distance": dists[top],
            "cluster": labels[top],
        }
    )


def primary_similarity(
    wf: WindowFeatures,
    n_clusters: int,
    reference_idx: int | None = None,
    top_k: int = 10,
    min_gap: int | None = None,
    robust: bool = False,
) -> SimilarityResult:
    X = _standardize_features(wf.features)
    ref_idx = len(X) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    vi = _cov_inv(X, robust=robust)
    ref = X[ref_idx]
    dists = np.array([mahalanobis(x, ref, vi) for x in X])

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)

    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=_build_top_df(dists, labels, wf.end_dates, ref_idx, gap, top_k),
        method="primary",
    )


def cosine_kmeans_similarity(
    wf: WindowFeatures,
    n_clusters: int,
    reference_idx: int | None = None,
    top_k: int = 10,
    min_gap: int | None = None,
    robust: bool = False,
) -> SimilarityResult:
    """Direction-based similarity: cosine distance between L2-normalized
    feature vectors, KMeans clustering on the same normalized space
    (spherical k-means via unit-vector centroids)."""
    X = _standardize_features(wf.features)
    Xn = normalize(X, norm="l2", axis=1)
    ref_idx = len(X) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    ref = Xn[ref_idx]
    # cosine distance = 1 - cosine similarity
    dists = 1.0 - Xn @ ref

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = model.fit_predict(Xn)

    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=_build_top_df(dists, labels, wf.end_dates, ref_idx, gap, top_k),
        method="cosine",
    )


def gmm_similarity(
    wf: WindowFeatures,
    n_clusters: int,
    reference_idx: int | None = None,
    top_k: int = 10,
    min_gap: int | None = None,
    robust: bool = False,
) -> SimilarityResult:
    """Euclidean distance in standardized feature space; soft-probabilistic
    clustering via a Gaussian Mixture Model. Hard labels are argmax of the
    posterior — the GMM log-likelihood is still the natural soft membership."""
    X = _standardize_features(wf.features)
    ref_idx = len(X) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    ref = X[ref_idx]
    dists = np.linalg.norm(X - ref, axis=1)

    cov_type = "diag"  # robust to high-d features; full covariance would overfit
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=cov_type,
        random_state=0,
        reg_covar=1e-4,
        n_init=3,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=_build_top_df(dists, labels, wf.end_dates, ref_idx, gap, top_k),
        method="gmm",
    )


def _sbd(a: np.ndarray, b: np.ndarray) -> float:
    """Shape-based distance = 1 - max normalized cross-correlation.
    Works on 1D; multivariate windows are handled by averaging SBD across series."""
    num = np.convolve(a, b[::-1], mode="full")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - num.max() / denom)


def _pairwise_sbd_to_reference(
    panel_slice: pd.DataFrame,
    window_size: int,
    end_dates: pd.DatetimeIndex,
    reference_idx: int,
) -> np.ndarray:
    values = panel_slice.to_numpy(dtype=float)
    n_windows = len(end_dates)
    d = values.shape[1]

    def window(i):
        return values[i : i + window_size]

    ref = window(reference_idx)
    out = np.zeros(n_windows)
    for i in range(n_windows):
        w = window(i)
        # average SBD across series
        vals = [_sbd(ref[:, j], w[:, j]) for j in range(d)]
        out[i] = float(np.mean(vals))
    return out


def secondary_similarity(
    wf: WindowFeatures,
    reference_idx: int | None = None,
    top_k: int = 10,
    min_gap: int | None = None,
    min_cluster_size: int | None = None,
) -> SimilarityResult:
    import hdbscan  # heavy import, lazy

    X = _standardize_features(wf.features)
    ref_idx = len(X) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    dists = _pairwise_sbd_to_reference(
        wf.panel_slice, wf.window_size, wf.end_dates, ref_idx
    )

    mcs = max(20, len(X) // 40) if min_cluster_size is None else min_cluster_size
    labels = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(X)

    idx = np.arange(len(X))
    mask = np.abs(idx - ref_idx) >= gap
    ranked = idx[mask][np.argsort(dists[mask])]
    top = ranked[:top_k]
    top_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(top) + 1),
            "end_date": wf.end_dates[top],
            "distance": dists[top],
            "cluster": labels[top],
        }
    )

    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=top_df,
        method="secondary",
    )


def embed_2d(
    wf: WindowFeatures,
    method: Literal["umap", "tsne", "pca"] = "umap",
    random_state: int = 0,
) -> np.ndarray:
    X = _standardize_features(wf.features)
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
    if method == "tsne":
        perp = min(30, max(5, len(X) // 30))
        return TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            random_state=random_state,
            learning_rate="auto",
        ).fit_transform(X)
    # umap
    import umap  # heavy import, lazy
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=random_state,
        metric="euclidean",
    )
    return reducer.fit_transform(X)
