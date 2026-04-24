"""Analog-based backtest.

Frame each of the top-K historical analogs as a single trade: if you had
bought (or tracked) the asset at the end of each analog window and held
for horizon H, what would the distribution of outcomes look like?

We report, per asset per horizon:
  - N (count of valid analogs)
  - mean, median, std of forward return / bps change
  - hit_rate (% positive for ret assets; share of favorable direction for bps)
  - sharpe, annualized from the horizon
  - min (worst case), max (best case)

Drawdown isn't well-defined for single-period snapshots, so we use
`min` as the worst-outcome proxy and report it explicitly.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .forward_returns import DEFAULT_ASSETS

# Horizons are in weeks; sqrt(periods_per_year / horizon_weeks) annualizes a single-period Sharpe
HORIZON_PERIODS_PER_YEAR = {
    "1m": 12.0,
    "3m": 4.0,
    "12m": 1.0,
}


def backtest_stats(
    fwd: pd.DataFrame,
    horizon_key: str,
    assets: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per-asset distributional stats across the K analog forward outcomes.

    `fwd` is one of the frames in fwd_by_horizon: rows = analog end_dates,
    cols = asset codes. `horizon_key` is "1m" | "3m" | "12m" for annualization.
    """
    assets = assets or DEFAULT_ASSETS
    periods_per_year = HORIZON_PERIODS_PER_YEAR.get(horizon_key, 4.0)

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
        sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan
        mn = float(x.min())
        mx = float(x.max())

        if kind == "ret":
            hit = float((x > 0).mean())
            hit_label = "win rate"
        else:
            # For yields/spreads, lower is generally favorable (rates falling,
            # spreads tightening). We report the share < 0 as "tightening rate".
            hit = float((x < 0).mean())
            hit_label = "tighten rate"

        rows.append({
            "asset": asset,
            "kind": kind,
            "N": int(len(x)),
            "mean": mean,
            "median": median,
            "std": std,
            "sharpe": sharpe,
            "min": mn,
            "max": mx,
            "hit_rate": hit,
            "hit_label": hit_label,
        })

    return pd.DataFrame(rows)


def format_backtest(stats: pd.DataFrame) -> pd.DataFrame:
    """Human-readable formatting: % for ret, bps for yield/spread."""
    out = stats.copy()
    def fmt(v, kind, precision=2):
        if pd.isna(v):
            return ""
        if kind == "ret":
            return f"{v * 100:.{precision}f}%"
        return f"{int(round(v))} bps"

    for col in ["mean", "median", "min", "max"]:
        out[col] = [fmt(v, k) for v, k in zip(stats[col], stats["kind"])]
    # std stays in native units (annualized-like) — express as % or bps too
    out["std"] = [
        f"{v * 100:.2f}%" if k == "ret" else f"{int(round(v))} bps"
        for v, k in zip(stats["std"], stats["kind"])
    ]
    out["sharpe"] = stats["sharpe"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    out["hit_rate"] = [
        f"{v * 100:.0f}% ({lbl})"
        for v, lbl in zip(stats["hit_rate"], stats["hit_label"])
    ]
    return out[["asset", "N", "mean", "median", "std", "sharpe", "min", "max", "hit_rate"]]
