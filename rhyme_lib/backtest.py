"""Analog-based diagnostics:

1. `backtest_stats` — *cross-sectional* summary of the K analog forward
   outcomes at today's reference window. Useful for gauging analog
   consensus. NOT a strategy Sharpe: it describes dispersion of K historical
   outcomes, not the return-per-risk of a trading strategy. For that, see
   `walk_forward_backtest` below.

2. `walk_forward_backtest` — honest walk-forward backtest. For every
   historical decision date T, we:
     a) use ONLY window features strictly before T - horizon (so every
        candidate analog's forward return is fully realized before T);
     b) re-fit the distance engine on past-only features (past-only
        StandardScaler, past-only LedoitWolf / MinCovDet for Mahalanobis);
     c) take the sign of the mean forward return across the available
        top-K past analogs as the position at T;
     d) realize the actual panel return from T to T+horizon;
     e) record strategy_return(T) = position × realized_return(T→T+H).
   Step size = horizon (non-overlapping trades). Early history allowed
   with fewer analogs — the signal just uses whatever's available.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .forward_returns import DEFAULT_ASSETS, DEFAULT_HORIZONS_WEEKS
from .similarity import (
    _cov_inv,
    _loglik_signature,
    _pairwise_mahalanobis,
    _sbd,
    _standardize_with_impute,
)

HORIZON_PERIODS_PER_YEAR = {"1m": 12.0, "3m": 4.0, "12m": 1.0}
HORIZON_PERIODS_BY_FREQ: dict[str, dict[str, int]] = {
    "M": {"1m": 1, "3m": 3, "12m": 12},
    "W": {"1m": 4, "3m": 13, "12m": 52},
    "D": {"1m": 21, "3m": 63, "12m": 252},
}


# ---------------------------------------------------------------------------
# Cross-sectional "today's analog outcomes" summary
# ---------------------------------------------------------------------------


def backtest_stats(
    fwd: pd.DataFrame,
    horizon_key: str,
    assets: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per-asset dispersion stats across the K analog forward outcomes at
    today's reference. Use this to read analog consensus — it is NOT a
    trading Sharpe (that lives in walk_forward_backtest)."""
    assets = assets or DEFAULT_ASSETS

    rows = []
    for asset, kind in assets.items():
        if asset not in fwd.columns:
            continue
        x = fwd[asset].dropna()
        if len(x) == 0:
            continue

        mean = float(x.mean())
        median = float(x.median())
        std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        mn = float(x.min())
        mx = float(x.max())

        if kind == "ret":
            hit = float((x > 0).mean())
            hit_label = "up"
        else:
            hit = float((x < 0).mean())
            hit_label = "down"

        rows.append({
            "asset": asset, "kind": kind, "N": int(len(x)),
            "mean": mean, "median": median, "std": std,
            "min": mn, "max": mx, "hit_rate": hit, "hit_label": hit_label,
        })
    return pd.DataFrame(rows)


def format_backtest(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.copy()

    def fmt(v, kind):
        if pd.isna(v):
            return ""
        return f"{v * 100:.2f}%" if kind == "ret" else f"{int(round(v))} bps"

    for col in ["mean", "median", "min", "max", "std"]:
        out[col] = [fmt(v, k) for v, k in zip(stats[col], stats["kind"])]
    out["agree"] = [
        f"{v * 100:.0f}% {lbl}"
        for v, lbl in zip(stats["hit_rate"], stats["hit_label"])
    ]
    return out[["asset", "N", "mean", "median", "std", "min", "max", "agree"]]


# ---------------------------------------------------------------------------
# Walk-forward strategy backtest (no look-ahead)
# ---------------------------------------------------------------------------


def _fwd_returns_asof(
    panel_daily: pd.DataFrame,
    ref_dates: pd.DatetimeIndex,
    horizon_weeks: int,
    assets: Mapping[str, str],
) -> pd.DataFrame:
    """Like forward_returns, but returns NaN when T + horizon is beyond the
    panel's available data (instead of the silent ffill that `reindex` does).

    Looks up prices via Series.asof so we get the last known value strictly
    at or before each target date — the safe, no-peek alternative to
    reindex(method='ffill') when the target may be outside the index.
    """
    days = horizon_weeks * 7
    out = pd.DataFrame(index=ref_dates)
    for asset, kind in assets.items():
        col = panel_daily.get(asset)
        if col is None:
            out[asset] = np.nan
            continue
        col = col.dropna().sort_index()
        if len(col) == 0:
            out[asset] = np.nan
            continue
        last = col.index.max()
        first = col.index.min()

        vals = []
        for d in ref_dates:
            target = d + pd.Timedelta(days=days)
            if target > last or d < first:
                vals.append(np.nan)
                continue
            now_v = col.asof(d)
            fwd_v = col.asof(target)
            if pd.isna(now_v) or pd.isna(fwd_v):
                vals.append(np.nan)
                continue
            if kind == "ret":
                if now_v > 0 and fwd_v > 0:
                    vals.append(float(np.log(fwd_v / now_v)))
                else:
                    vals.append(np.nan)
            else:
                vals.append(float((fwd_v - now_v) * 100.0))
        out[asset] = vals
    return out


def _distances_past_only(
    wf_features: np.ndarray,
    wf_panel_slice: pd.DataFrame,
    window_size: int,
    T_idx: int,
    valid_end: int,  # inclusive; analog indices in [0, valid_end]
    method: str,
    robust: bool,
) -> np.ndarray:
    """Distance from window T_idx to every candidate analog in [0, valid_end],
    using ONLY features in that past range to fit scalers / covariances.

    NaN-aware: ``wf_features`` may contain NaNs (a column inactive in a
    given window). Standardize NaN-aware on past-only stats; the
    Mahalanobis distance uses the per-pair active-mask intersection so a
    window with limited coverage isn't artificially close to anything.
    """
    past_X = wf_features[: valid_end + 1]
    ref = wf_features[T_idx]

    # Past-only standardization, NaN-aware.
    past_Xs, means, stds, valid = _standardize_with_impute(past_X)
    past_nan_mask = (~np.isnan(past_X))[:, valid]
    past_Xs = past_Xs[:, valid]
    ref_active = ~np.isnan(ref)
    # Re-impute ref against past-only means (NaN where neither column is active in past).
    ref_s = np.where(np.isnan(ref), 0.0, (ref - means) / stds)
    ref_s = ref_s[valid]
    ref_mask = ref_active[valid]

    if method == "primary":
        vi = _cov_inv(past_Xs, robust=robust)
        return _pairwise_mahalanobis(past_Xs, past_nan_mask, ref_s, ref_mask, vi)
    if method == "cosine":
        n = past_Xs.shape[0]
        out = np.zeros(n)
        for i in range(n):
            common = past_nan_mask[i] & ref_mask
            if common.sum() < 5:
                out[i] = np.nan
                continue
            a = ref_s[common]
            b = past_Xs[i, common]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            out[i] = 1.0 - (a @ b) / denom if denom > 0 else 1.0
        return out
    if method == "gmm":
        # Fit GMM on past-only; distance = Euclidean on per-component weighted
        # log-likelihood signatures (matches live gmm_similarity).
        from sklearn.mixture import GaussianMixture
        n_comp = max(2, min(4, past_Xs.shape[0] // 20))
        gmm = GaussianMixture(
            n_components=n_comp, covariance_type="diag",
            random_state=0, reg_covar=1e-4, n_init=1,
        )
        gmm.fit(past_Xs)
        sig_all = _loglik_signature(gmm, np.vstack([past_Xs, ref_s.reshape(1, -1)]))
        sig_past = sig_all[:-1]
        # Clip and standardize using PAST-ONLY stats
        lo = np.quantile(sig_past, 0.01, axis=0)
        hi = np.quantile(sig_past, 0.99, axis=0)
        sig_past_c = np.clip(sig_past, lo, hi)
        ref_c = np.clip(sig_all[-1], lo, hi)
        mu = sig_past_c.mean(axis=0)
        sd = sig_past_c.std(axis=0) + 1e-8
        sig_past_std = (sig_past_c - mu) / sd
        ref_std = (ref_c - mu) / sd
        return np.linalg.norm(sig_past_std - ref_std, axis=1)
    if method == "secondary":
        # SBD over the panel slice (shape-based, window-by-window).
        ref_block = wf_panel_slice.iloc[T_idx : T_idx + window_size].to_numpy(dtype=float)
        out = np.zeros(valid_end + 1)
        for i in range(valid_end + 1):
            w = wf_panel_slice.iloc[i : i + window_size].to_numpy(dtype=float)
            d = w.shape[1]
            vals = []
            for j in range(d):
                v = _sbd(ref_block[:, j], w[:, j])
                if np.isfinite(v):
                    vals.append(v)
            out[i] = float(np.mean(vals)) if vals else np.nan
        return out
    raise ValueError(f"unknown method: {method}")


def walk_forward_backtest(
    wf_features: np.ndarray,
    wf_end_dates: pd.DatetimeIndex,
    wf_panel_slice: pd.DataFrame,
    window_size: int,
    panel_daily: pd.DataFrame,
    freq: str,
    method: str,
    robust: bool,
    top_k: int,
    horizon_key: str,
    assets: Mapping[str, str] | None = None,
    min_analogs: int = 1,
) -> dict:
    """Run a true walk-forward backtest and return per-asset strategy stats
    plus the per-trade record and per-asset equity curves.

    At each T we use only pre-T information (features[:T_idx+1-horizon]);
    the position is sign(mean analog forward return); the strategy return
    is position * realized(T -> T+horizon). Trades are non-overlapping
    (step = horizon).
    """
    assets = assets or DEFAULT_ASSETS
    h_periods = HORIZON_PERIODS_BY_FREQ.get(freq, HORIZON_PERIODS_BY_FREQ["M"])[horizon_key]
    h_weeks = DEFAULT_HORIZONS_WEEKS[horizon_key]
    periods_per_year = HORIZON_PERIODS_PER_YEAR[horizon_key]

    n_windows = len(wf_end_dates)
    # Need at least `min_analogs` past windows with forward returns realized.
    # valid_end(T_idx) = T_idx - h_periods  (inclusive)
    # So first usable T_idx satisfies T_idx - h_periods >= min_analogs - 1
    start_idx = h_periods + max(min_analogs, 1)
    step = max(1, h_periods)

    # Only go as far as T + h_periods is representable in the window grid; the
    # realized return check using panel_daily handles the actual data bound.
    end_idx = n_windows
    test_indices = list(range(start_idx, end_idx, step))
    if not test_indices:
        return {"trades": pd.DataFrame(), "stats": pd.DataFrame(), "equity": {}}

    trade_records: list[dict] = []
    for T_idx in test_indices:
        T = wf_end_dates[T_idx]
        valid_end = T_idx - h_periods  # inclusive
        if valid_end < 0:
            continue

        dists = _distances_past_only(
            wf_features, wf_panel_slice, window_size,
            T_idx, valid_end, method, robust,
        )
        n_past = len(dists)
        top_n = int(min(top_k, n_past))
        if top_n < min_analogs:
            continue
        top_indices = np.argsort(dists)[:top_n]
        analog_dates = pd.DatetimeIndex([wf_end_dates[i] for i in top_indices])

        # Analog forward returns — fully realized before T by construction.
        analog_fwd = _fwd_returns_asof(panel_daily, analog_dates, h_weeks, assets)
        signals = analog_fwd.mean(skipna=True)

        realized_fwd = _fwd_returns_asof(panel_daily, pd.DatetimeIndex([T]), h_weeks, assets)
        if realized_fwd.iloc[0].isna().all():
            # T + horizon is past the end of the panel; nothing to evaluate.
            continue
        realized = realized_fwd.iloc[0]

        rec: dict = {"date": T, "n_analogs": top_n}
        for asset in assets:
            sig = signals.get(asset, np.nan)
            real = realized.get(asset, np.nan)
            if pd.isna(sig) or pd.isna(real):
                pos = 0.0
                strat: float = np.nan
            else:
                pos = 1.0 if sig > 0 else (-1.0 if sig < 0 else 0.0)
                strat = pos * real
            rec[f"signal_{asset}"] = sig
            rec[f"pos_{asset}"] = pos
            rec[f"realized_{asset}"] = real
            rec[f"strategy_{asset}"] = strat
        trade_records.append(rec)

    trades = pd.DataFrame(trade_records)
    if trades.empty:
        return {"trades": trades, "stats": pd.DataFrame(), "equity": {}}

    stats_rows: list[dict] = []
    equity: dict[str, pd.Series] = {}
    for asset, kind in assets.items():
        col = f"strategy_{asset}"
        if col not in trades.columns:
            continue
        pair = trades[["date", col]].dropna()
        if len(pair) == 0:
            continue
        s = pair[col].astype(float)
        idx = pd.DatetimeIndex(pair["date"].values)

        mean = float(s.mean())
        std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        sharpe = float((mean / std) * np.sqrt(periods_per_year)) if std > 0 else np.nan
        hit = float((s > 0).mean())

        if kind == "ret":
            equity_curve = (1.0 + s.values).cumprod()
            cum = float(equity_curve[-1] - 1.0)
            running_max = np.maximum.accumulate(equity_curve)
            dd = equity_curve / running_max - 1.0
            max_dd = float(dd.min())
        else:
            equity_curve = s.values.cumsum()
            cum = float(equity_curve[-1])
            running_max = np.maximum.accumulate(equity_curve)
            dd = equity_curve - running_max
            max_dd = float(dd.min())

        equity[asset] = pd.Series(equity_curve, index=idx, name=asset)
        stats_rows.append({
            "asset": asset, "kind": kind, "N": int(len(s)),
            "mean": mean, "std": std, "sharpe": sharpe, "hit_rate": hit,
            "cum": cum, "max_dd": max_dd,
        })

    return {"trades": trades, "stats": pd.DataFrame(stats_rows), "equity": equity}


def format_walk_forward(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return stats
    out = stats.copy()

    def fmt(v, kind):
        if pd.isna(v):
            return ""
        return f"{v * 100:.2f}%" if kind == "ret" else f"{int(round(v))} bps"

    out["mean"] = [fmt(v, k) for v, k in zip(stats["mean"], stats["kind"])]
    out["std"] = [fmt(v, k) for v, k in zip(stats["std"], stats["kind"])]
    out["cum"] = [
        f"{v * 100:.1f}%" if k == "ret" else f"{int(round(v))} bps"
        for v, k in zip(stats["cum"], stats["kind"])
    ]
    out["max_dd"] = [
        f"{v * 100:.1f}%" if k == "ret" else f"{int(round(v))} bps"
        for v, k in zip(stats["max_dd"], stats["kind"])
    ]
    out["sharpe"] = stats["sharpe"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    out["hit_rate"] = stats["hit_rate"].apply(
        lambda v: f"{v * 100:.0f}%" if pd.notna(v) else ""
    )
    return out[["asset", "N", "mean", "std", "sharpe", "hit_rate", "cum", "max_dd"]]
