"""Stationarize, z-score, and theme-aggregate the panel.

Pipeline:
  raw daily panel  -> resample to target freq
                   -> per-series stationarization (level / log-diff / bps-diff / yoy)
                   -> rolling (or expanding) z-score on each transformed series
                   -> theme aggregation (equal-weight z-scores within bucket)

Outputs:
  features_df  : per-series transformed+z-scored columns (wide)
  themes_df    : one column per bucket (mean of that bucket's z-scored members)
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

TargetFreq = Literal["D", "W", "M"]
ZMode = Literal["rolling", "expanding"]

DEFAULT_ROLLING_YEARS = 20
DEFAULT_MIN_YEARS = 10


def resample_panel(panel: pd.DataFrame, freq: TargetFreq) -> pd.DataFrame:
    """Resample daily panel to target freq using last observation of the period."""
    rule_map = {"D": "D", "W": "W-FRI", "M": "ME"}
    rule = rule_map[freq]
    out = panel.resample(rule).last()
    out = out.ffill(limit=6)  # fill small gaps only
    return out


def _stationarize_col(
    s: pd.Series,
    transform: str,
    freq: TargetFreq,
) -> pd.Series:
    """Apply the per-series stationarization recipe. Returns same-length series
    with NaNs at the head where the transform requires history."""
    periods_3m, periods_12m = _lookback_periods(freq)

    if transform == "level":
        return s.astype(float)

    if transform == "log_diff":
        # 3m log-diff ("change"); we retain just one dimension per series to
        # keep the feature space manageable — 3m is the primary signal.
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(s.where(s > 0))
        return (lg - lg.shift(periods_3m)).rename(s.name)

    if transform == "bps_diff":
        return (s - s.shift(periods_3m)).rename(s.name)

    if transform == "yoy_log":
        smooth = s.rolling(periods_3m, min_periods=1).mean() if freq == "W" else s
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(smooth.where(smooth > 0))
        return (lg - lg.shift(periods_12m)).rename(s.name)

    if transform == "already_yoy":
        return s.astype(float)

    raise ValueError(f"unknown transform {transform}")


def _lookback_periods(freq: TargetFreq) -> tuple[int, int]:
    """Return (3-month, 12-month) lookbacks in resampled periods."""
    if freq == "M":
        return 3, 12
    if freq == "W":
        return 13, 52
    return 63, 252  # D (business days)


MAD_TO_STD = 1.4826  # scale factor to make MAD comparable to std for Gaussian data


def _mad(x: pd.Series) -> float:
    """Median absolute deviation, scaled to match std on Gaussian data."""
    med = x.median()
    return MAD_TO_STD * (x - med).abs().median()


def _z_score(
    df: pd.DataFrame,
    freq: TargetFreq,
    mode: ZMode = "rolling",
    rolling_years: int = DEFAULT_ROLLING_YEARS,
    min_years: int = DEFAULT_MIN_YEARS,
    robust: bool = False,
) -> pd.DataFrame:
    periods_per_year = {"D": 252, "W": 52, "M": 12}[freq]
    window = rolling_years * periods_per_year
    min_periods = min_years * periods_per_year

    if mode == "rolling":
        roller = df.rolling(window, min_periods=min_periods)
    else:
        roller = df.expanding(min_periods=min_periods)

    if robust:
        center = roller.median()
        scale = roller.apply(_mad_np, raw=True)
    else:
        center = roller.mean()
        scale = roller.std()

    out = ((df - center) / scale.replace(0, np.nan)).rename(columns=lambda c: f"{c}_z")
    if robust:
        # Winsorize extreme robust z values so a single shock doesn't dominate feature moments
        out = out.clip(lower=-5.0, upper=5.0)
    return out


def _mad_np(x: np.ndarray) -> float:
    """Fast MAD for rolling.apply (numeric-only, pre-scaled to match std)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(MAD_TO_STD * np.median(np.abs(x - med)))


def transform_and_zscore(
    panel: pd.DataFrame,
    transform_map: Mapping[str, str],
    freq: TargetFreq = "W",
    mode: ZMode = "rolling",
    rolling_years: int = DEFAULT_ROLLING_YEARS,
    min_years: int = DEFAULT_MIN_YEARS,
    robust: bool = False,
) -> pd.DataFrame:
    resampled = resample_panel(panel, freq)

    cols: list[pd.Series] = []
    for col in resampled.columns:
        t = transform_map.get(col, "level")
        cols.append(_stationarize_col(resampled[col], t, freq))
    stationarized = pd.concat(cols, axis=1)

    if robust:
        # Per-column winsorization at 1%/99% quantiles before z-scoring to cap outliers.
        lo = stationarized.quantile(0.01)
        hi = stationarized.quantile(0.99)
        stationarized = stationarized.clip(lower=lo, upper=hi, axis=1)

    return _z_score(stationarized, freq, mode, rolling_years, min_years, robust=robust)


def theme_aggregate(
    zdf: pd.DataFrame,
    bucket_map: Mapping[str, str],
    robust: bool = False,
) -> pd.DataFrame:
    """Equal-weight average of z-scored columns within each bucket. Expects
    columns in zdf named "<code>_z"; bucket_map is {code: bucket}.
    If `robust`, uses median instead of mean."""
    buckets: dict[str, list[str]] = {}
    for zcol in zdf.columns:
        code = zcol[:-2] if zcol.endswith("_z") else zcol
        b = bucket_map.get(code)
        if b is None:
            continue
        buckets.setdefault(b, []).append(zcol)
    reducer = "median" if robust else "mean"
    return pd.DataFrame(
        {b: getattr(zdf[cols], reducer)(axis=1) for b, cols in buckets.items()},
        index=zdf.index,
    )


def infer_transforms(panel: pd.DataFrame) -> dict[str, str]:
    """Heuristic for user-uploaded panels: pick a transform per column based
    on its statistical profile. Only used when panel is not the default one."""
    out: dict[str, str] = {}
    for c in panel.columns:
        s = panel[c].dropna()
        if len(s) < 30:
            out[c] = "level"
            continue
        # Values that look like rates (small magnitudes, could be +/-) -> level
        if s.abs().median() < 25 and s.min() > -50 and s.max() < 50:
            out[c] = "level"
        elif (s > 0).all() and s.median() > 10:
            out[c] = "log_diff"
        else:
            out[c] = "level"
    return out
