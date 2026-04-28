"""Similarity and clustering engine.

Primary methodology: Mahalanobis distance (Ledoit-Wolf shrunk covariance)
on the window feature matrix, plus Ward hierarchical clustering. This is
the Kritzman-Page-Turkington "turbulence" formulation extended to a richer
feature set.

Secondary methodology: Shape-based distance (k-Shape / SBD) on the raw
z-scored panel slice, plus HDBSCAN density-based clustering.

Embeddings for the scatter view: UMAP (primary), t-SNE, PCA (toggles).

NaN-tolerant rewrite (2026-04).  The window feature matrix now contains
NaNs by design — a column inactive in a window has NaN moments and NaN
correlation entries.  All distance computations use a per-row valid
mask: when comparing window A to window B, only features active in
*both* are kept.  The covariance / scaler is fit on each feature's
column, ignoring NaN rows; if a feature has no valid entries it is
dropped from the active basis entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from .features import WindowFeatures

Method = Literal["primary", "secondary", "cosine", "gmm"]


@dataclass
class SimilarityResult:
    labels: np.ndarray              # cluster label per window
    distances: np.ndarray           # distance from each window to reference
    reference_idx: int
    top_analogs: pd.DataFrame       # end_date, distance, cluster, rank
    method: Method
    cluster_centroids: np.ndarray | None = None  # (k, d_imputed)
    feature_means: np.ndarray | None = None      # (d,) — used for re-imputation
    feature_stds: np.ndarray | None = None       # (d,)


# ---------------------------------------------------------------------------
# NaN-aware standardization & impute
# ---------------------------------------------------------------------------


def _column_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std (NaN-aware), with std=1 for empty columns."""
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
    return means, stds


def _standardize_with_impute(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize NaN-aware; remaining NaNs imputed at 0 (≡ column mean
    in z-space).  Returns (Xs, means, stds, valid_mask).

    Columns that are entirely NaN are dropped from the valid feature set.
    """
    means, stds = _column_stats(X)
    valid = (~np.isnan(X)).any(axis=0)
    Xs = (X - means) / stds
    Xs = np.where(np.isnan(Xs), 0.0, Xs)
    Xs = np.where(np.isfinite(Xs), Xs, 0.0)
    return Xs, means, stds, valid


def _cov_inv(X: np.ndarray, robust: bool = False) -> np.ndarray:
    if robust:
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


def _pairwise_mahalanobis(
    X: np.ndarray,
    nan_mask: np.ndarray,
    ref: np.ndarray,
    ref_mask: np.ndarray,
    vi_full: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance from ref to each row in X, on the per-pair
    intersection of active features.

    Args:
        X: (n, d) standardized features (NaN-imputed at 0).
        nan_mask: (n, d) True where the original feature was active.
        ref: (d,) reference (imputed).
        ref_mask: (d,) reference active mask.
        vi_full: (d, d) Mahalanobis metric on the full feature space.
    """
    n, d = X.shape
    out = np.zeros(n)
    for i in range(n):
        common = nan_mask[i] & ref_mask
        k = int(common.sum())
        if k < 5:
            out[i] = np.nan
            continue
        # Subset metric: pull the (k, k) block of vi_full.  This is an
        # approximation — strictly we'd recompute Σ^{-1} on the subset —
        # but on a global Ledoit-Wolf basis it's well-conditioned and
        # avoids n × d² cost per pair.
        idx = np.where(common)[0]
        diff = X[i, idx] - ref[idx]
        sub_vi = vi_full[np.ix_(idx, idx)]
        out[i] = float(np.sqrt(max(diff @ sub_vi @ diff, 0.0)))
    return out


def _build_top_df(
    dists: np.ndarray,
    labels: np.ndarray,
    end_dates: pd.DatetimeIndex,
    ref_idx: int,
    gap: int,
    top_k: int,
) -> pd.DataFrame:
    idx = np.arange(len(dists))
    finite = np.isfinite(dists)
    mask = (np.abs(idx - ref_idx) >= gap) & finite
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
    Xs, means, stds, valid = _standardize_with_impute(wf.features)
    nan_mask = ~np.isnan(wf.features)

    # Drop columns with no valid entries anywhere.
    Xs = Xs[:, valid]
    nan_mask = nan_mask[:, valid]
    means = means[valid]
    stds = stds[valid]

    ref_idx = len(Xs) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    vi_full = _cov_inv(Xs, robust=robust)
    dists = _pairwise_mahalanobis(
        Xs, nan_mask, Xs[ref_idx], nan_mask[ref_idx], vi_full,
    )

    # Clusters: AgglomerativeClustering doesn't support NaN, but Xs is
    # already NaN-imputed at the column mean — still meaningful in z-space.
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(Xs)

    # Cluster centroids in standardized space (for walk-forward labeling).
    centroids = np.vstack(
        [Xs[labels == c].mean(axis=0) for c in sorted(np.unique(labels))]
    )

    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=_build_top_df(dists, labels, wf.end_dates, ref_idx, gap, top_k),
        method="primary",
        cluster_centroids=centroids,
        feature_means=means,
        feature_stds=stds,
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
    Xs, means, stds, valid = _standardize_with_impute(wf.features)
    nan_mask = ~np.isnan(wf.features)
    Xs = Xs[:, valid]
    nan_mask = nan_mask[:, valid]
    means = means[valid]
    stds = stds[valid]

    Xn = normalize(Xs, norm="l2", axis=1)
    ref_idx = len(Xs) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    # Cosine on per-pair intersection — re-normalize the masked subspace.
    n, d = Xs.shape
    ref = Xs[ref_idx]
    ref_active = nan_mask[ref_idx]
    dists = np.zeros(n)
    for i in range(n):
        common = nan_mask[i] & ref_active
        if common.sum() < 5:
            dists[i] = np.nan
            continue
        a = ref[common]
        b = Xs[i, common]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            dists[i] = 1.0
            continue
        dists[i] = float(1.0 - (a @ b) / denom)

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = model.fit_predict(Xn)

    centroids = np.vstack(
        [Xn[labels == c].mean(axis=0) for c in sorted(np.unique(labels))]
    )
    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=_build_top_df(dists, labels, wf.end_dates, ref_idx, gap, top_k),
        method="cosine",
        cluster_centroids=centroids,
        feature_means=means,
        feature_stds=stds,
    )


def _loglik_signature(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """Per-component weighted log-likelihood for each sample: log[π_k N(x|μ_k,Σ_k)].
    Shape (n_samples, n_components)."""
    return gmm._estimate_weighted_log_prob(X)


def gmm_similarity(
    wf: WindowFeatures,
    n_clusters: int,
    reference_idx: int | None = None,
    top_k: int = 10,
    min_gap: int | None = None,
    robust: bool = False,
) -> SimilarityResult:
    """Soft-probabilistic clustering via a Gaussian Mixture Model. Distance is
    Euclidean distance between per-component log-likelihood signatures."""
    Xs, means, stds, valid = _standardize_with_impute(wf.features)
    Xs = Xs[:, valid]
    means = means[valid]
    stds = stds[valid]
    ref_idx = len(Xs) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    cov_type = "diag"
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=cov_type,
        random_state=0,
        reg_covar=1e-4,
        n_init=3,
    )
    gmm.fit(Xs)
    sig = _loglik_signature(gmm, Xs)
    lo = np.quantile(sig, 0.01, axis=0)
    hi = np.quantile(sig, 0.99, axis=0)
    sig = np.clip(sig, lo, hi)
    sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)
    dists = np.linalg.norm(sig - sig[ref_idx], axis=1)
    labels = gmm.predict(Xs)

    centroids = np.vstack(
        [Xs[labels == c].mean(axis=0) for c in sorted(np.unique(labels))]
    )
    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=_build_top_df(dists, labels, wf.end_dates, ref_idx, gap, top_k),
        method="gmm",
        cluster_centroids=centroids,
        feature_means=means,
        feature_stds=stds,
    )


def _sbd(a: np.ndarray, b: np.ndarray) -> float:
    """Shape-based distance = 1 - max normalized cross-correlation."""
    # Drop any NaNs pairwise to keep convolution numerically meaningful.
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return 1.0
    a = a[mask]
    b = b[mask]
    num = np.convolve(a, b[::-1], mode="full")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - num.max() / denom)


def _pairwise_sbd_to_reference(
    panel_slice: pd.DataFrame,
    window_size: int,
    end_dates: pd.DatetimeIndex,
    end_dates_full: pd.DatetimeIndex,
    reference_idx: int,
) -> np.ndarray:
    """SBD over the panel slice, NaN-tolerant, restricted to the kept windows."""
    values = panel_slice.to_numpy(dtype=float)
    panel_dates = panel_slice.index
    n_windows = len(end_dates)
    d = values.shape[1]

    # Map each kept end_date to its panel-relative window start.
    end_to_start: dict[pd.Timestamp, int] = {}
    for ed in end_dates:
        pos = panel_dates.get_loc(ed)
        end_to_start[ed] = pos - window_size + 1

    ref_start = end_to_start[end_dates[reference_idx]]
    ref = values[ref_start : ref_start + window_size]
    out = np.zeros(n_windows)
    for i, ed in enumerate(end_dates):
        s = end_to_start[ed]
        w = values[s : s + window_size]
        vals = []
        for j in range(d):
            v = _sbd(ref[:, j], w[:, j])
            if np.isfinite(v):
                vals.append(v)
        out[i] = float(np.mean(vals)) if vals else np.nan
    return out


def secondary_similarity(
    wf: WindowFeatures,
    reference_idx: int | None = None,
    top_k: int = 10,
    min_gap: int | None = None,
    min_cluster_size: int | None = None,
) -> SimilarityResult:
    import hdbscan  # heavy import, lazy

    Xs, means, stds, valid = _standardize_with_impute(wf.features)
    Xs = Xs[:, valid]
    means = means[valid]
    stds = stds[valid]
    ref_idx = len(Xs) - 1 if reference_idx is None else reference_idx
    gap = wf.window_size if min_gap is None else min_gap

    dists = _pairwise_sbd_to_reference(
        wf.panel_slice, wf.window_size, wf.end_dates, wf.end_dates, ref_idx,
    )

    mcs = max(20, len(Xs) // 40) if min_cluster_size is None else min_cluster_size
    labels = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(Xs)

    idx = np.arange(len(Xs))
    finite = np.isfinite(dists)
    mask = (np.abs(idx - ref_idx) >= gap) & finite
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

    centroids = np.vstack(
        [Xs[labels == c].mean(axis=0) for c in sorted(np.unique(labels))]
    ) if len(np.unique(labels)) > 0 else None

    return SimilarityResult(
        labels=labels,
        distances=dists,
        reference_idx=ref_idx,
        top_analogs=top_df,
        method="secondary",
        cluster_centroids=centroids,
        feature_means=means,
        feature_stds=stds,
    )


def embed_2d(
    wf: WindowFeatures,
    method: Literal["umap", "tsne", "pca"] = "umap",
    random_state: int = 0,
) -> np.ndarray:
    Xs, _, _, valid = _standardize_with_impute(wf.features)
    Xs = Xs[:, valid]
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(Xs)
    if method == "tsne":
        perp = min(30, max(5, len(Xs) // 30))
        return TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            random_state=random_state,
            learning_rate="auto",
        ).fit_transform(Xs)
    import umap  # heavy import, lazy
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=random_state,
        metric="euclidean",
    )
    return reducer.fit_transform(Xs)


# ---------------------------------------------------------------------------
# Bayesian regime probabilities
# ---------------------------------------------------------------------------


def regime_probabilities(
    Xs: np.ndarray,
    nan_mask: np.ndarray,
    cluster_centroids: np.ndarray,
    cluster_labels: np.ndarray,
    vi_full: np.ndarray,
    ref_idx: int,
    tau: float | None = None,
) -> np.ndarray:
    """Softmax over regimes weighted by inverse Mahalanobis distance to
    each cluster centroid: p_k ∝ exp(-d_k^2 / τ).  Uses the same per-pair
    NaN-aware Mahalanobis as ``primary_similarity``.

    Args:
        Xs: (n, d) standardized features.
        nan_mask: (n, d) original-active mask.
        cluster_centroids: (k, d).
        cluster_labels: (n,) cluster labels (used to set τ if not given).
        vi_full: (d, d) Mahalanobis metric.
        ref_idx: index of reference window.
        tau: temperature.  If None, set to the median squared
            within-cluster Mahalanobis distance — i.e. typical
            within-cluster spread.

    Returns: (k,) probability vector summing to 1.
    """
    ref = Xs[ref_idx]
    ref_mask = nan_mask[ref_idx]

    # Distance from ref to each centroid.  Centroids have no NaNs (they're
    # mean of imputed Xs); use ref's mask.
    k = cluster_centroids.shape[0]
    d_ref = np.zeros(k)
    for c in range(k):
        diff = ref - cluster_centroids[c]
        diff = diff[ref_mask]
        sub_vi = vi_full[np.ix_(np.where(ref_mask)[0], np.where(ref_mask)[0])]
        d_ref[c] = float(np.sqrt(max(diff @ sub_vi @ diff, 0.0)))

    if tau is None:
        # Typical within-cluster squared distance.
        within = []
        for c in sorted(np.unique(cluster_labels)):
            mem = Xs[cluster_labels == c]
            cen = cluster_centroids[c]
            for x in mem:
                diff = x - cen
                within.append(diff @ vi_full @ diff)
        tau = max(float(np.median(within)) if within else 1.0, 1e-6)

    z = -(d_ref ** 2) / tau
    z -= z.max()  # numerical stability
    p = np.exp(z)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Block-bootstrap null distribution
# ---------------------------------------------------------------------------


def block_bootstrap_null_distance(
    Xs: np.ndarray,
    nan_mask: np.ndarray,
    vi_full: np.ndarray,
    ref_idx: int,
    block_size: int = 24,
    n_reps: int = 1000,
    rng_seed: int = 0,
    min_gap: int = 12,
) -> np.ndarray:
    """Block-permute the rows of Xs and record the best-analog Mahalanobis
    distance to the reference under each permutation.  Returns an
    n_reps-vector of "best (smallest) distance under random data" — the
    null against which today's best-analog distance can be compared.
    """
    rng = np.random.default_rng(rng_seed)
    n = Xs.shape[0]
    ref = Xs[ref_idx]
    ref_active = nan_mask[ref_idx]
    out = np.zeros(n_reps)

    n_blocks = max(1, n // block_size)
    for r in range(n_reps):
        order = rng.permutation(n_blocks)
        idx: list[int] = []
        for b in order:
            start = b * block_size
            idx.extend(range(start, min(start + block_size, n)))
        idx = np.asarray(idx[:n], dtype=int)
        Xp = Xs[idx]
        Mp = nan_mask[idx]
        # Best-analog distance under the permutation, excluding a small gap
        # around ref_idx (preserve the original ref).
        local = _pairwise_mahalanobis(Xp, Mp, ref, ref_active, vi_full)
        valid_idx = np.arange(n)
        keep = (np.abs(valid_idx - ref_idx) >= min_gap) & np.isfinite(local)
        if keep.any():
            out[r] = float(np.min(local[keep]))
        else:
            out[r] = np.nan
    return out
