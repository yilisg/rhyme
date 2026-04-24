"""Contextual regime labels from cluster mean theme z-scores.

Two labeler flavors:

**Macro** (growth × inflation grid; monetary modifier) — appropriate when
the clustering is driven by macro series (growth + inflation buckets).

                      inflation > +.2   neutral |I| <= .2   inflation < -.2
  growth > +.2    ->  Reflation          Expansion           Goldilocks
  neutral         ->  Inflationary       Neutral             Disinflation
  growth < -.2    ->  Stagflation        Slowdown            Deflationary bust

Modifier from monetary theme:
  monetary z > +.5  -> " (risk-off)"
  monetary z < -.5  -> " (risk-on)"

**Market** (monetary × sentiment grid; VIX modifier) — appropriate when
the clustering is driven by monetary + sentiment buckets.

                      sentiment > +.25   neutral |S| <= .25   sentiment < -.25
  monetary < -.25 ->  Melt-up            Risk-on              Recovery
  neutral         ->  Bullish            Sideways             Cautious
  monetary > +.25 ->  Tightening peak    Risk-off             Crisis

Modifier from VIX z-score (if available in the panel):
  vix z > +.7  -> " (high vol)"
  vix z < -.7  -> " (calm)"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd

GROWTH = "growth"
INFL = "inflation"
FIN = "monetary"
SENT = "sentiment"

# Macro thresholds — tightened from 0.30 to 0.20 so clusters with small but
# directional theme differences still get distinguished labels (helpful when
# the clustered sample covers a single era, e.g. post-2018 macro).
MACRO_THRESHOLD = 0.15
MACRO_RISK_MOD = 0.50

# Market thresholds
MARKET_THRESHOLD = 0.25
MARKET_VOL_MOD = 0.70

Mode = Literal["macro", "market"]


@dataclass
class RegimeLabel:
    cluster: int
    label: str
    growth_z: float
    inflation_z: float
    financial_z: float
    sentiment_z: float
    vix_z: float
    n_windows: int


def _macro_base(g: float, i: float) -> str:
    t = MACRO_THRESHOLD
    if g > t and i > t:
        return "Reflation"
    if g > t and abs(i) <= t:
        return "Expansion"
    if g > t and i < -t:
        return "Goldilocks"
    if abs(g) <= t and i > t:
        return "Inflationary"
    if abs(g) <= t and abs(i) <= t:
        return "Neutral"
    if abs(g) <= t and i < -t:
        return "Disinflation"
    if g < -t and i > t:
        return "Stagflation"
    if g < -t and abs(i) <= t:
        return "Slowdown"
    return "Deflationary bust"


def _macro_suffix(fin_z: float) -> str:
    if fin_z > MACRO_RISK_MOD:
        return " (risk-off)"
    if fin_z < -MACRO_RISK_MOD:
        return " (risk-on)"
    return ""


def _market_base(m: float, s: float) -> str:
    """Monetary-financial (x) × sentiment (y) grid for market mode."""
    t = MARKET_THRESHOLD
    # Easy monetary (low z)
    if m < -t and s > t:
        return "Melt-up"
    if m < -t and abs(s) <= t:
        return "Risk-on"
    if m < -t and s < -t:
        return "Recovery"
    # Neutral monetary
    if abs(m) <= t and s > t:
        return "Bullish"
    if abs(m) <= t and abs(s) <= t:
        return "Sideways"
    if abs(m) <= t and s < -t:
        return "Cautious"
    # Tight / stressed monetary (high z)
    if m > t and s > t:
        return "Tightening peak"
    if m > t and abs(s) <= t:
        return "Risk-off"
    return "Crisis"


def _market_suffix(vix_z: float) -> str:
    if not np.isfinite(vix_z):
        return ""
    if vix_z > MARKET_VOL_MOD:
        return " (high vol)"
    if vix_z < -MARKET_VOL_MOD:
        return " (calm)"
    return ""


def label_clusters(
    themes: pd.DataFrame,
    labels: np.ndarray,
    end_dates: pd.DatetimeIndex,
    mode: Mode = "macro",
    robust: bool = False,
    individual_z: pd.DataFrame | None = None,
) -> list[RegimeLabel]:
    """Compute per-cluster mean theme z-scores and assign a mode-specific
    label. Themes must contain at least growth, inflation, monetary, sentiment
    (missing columns are treated as zero). If `individual_z` is provided and
    contains `vix_z`, the market-mode modifier is based on cluster-mean VIX.
    `mode="macro"` uses the growth × inflation grid; `mode="market"` uses
    the monetary × sentiment grid."""
    aligned = themes.reindex(end_dates)
    for required_col in (GROWTH, INFL, FIN, SENT):
        if required_col not in aligned.columns:
            aligned[required_col] = np.nan

    vix_aligned = None
    if individual_z is not None and "vix_z" in individual_z.columns:
        vix_aligned = individual_z["vix_z"].reindex(end_dates)

    reducer = np.nanmedian if robust else np.nanmean
    out: list[RegimeLabel] = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        g = float(reducer(aligned.loc[mask, GROWTH].values))
        i = float(reducer(aligned.loc[mask, INFL].values))
        f = float(reducer(aligned.loc[mask, FIN].values))
        s_ = float(reducer(aligned.loc[mask, SENT].values))
        vix_z = (
            float(reducer(vix_aligned.loc[mask].values)) if vix_aligned is not None else np.nan
        )

        if cluster == -1:
            lbl = "Unclustered"
        elif mode == "market":
            base = _market_base(f, s_)
            lbl = base + _market_suffix(vix_z)
        else:
            base = _macro_base(g, i)
            lbl = base + _macro_suffix(f)

        out.append(
            RegimeLabel(
                cluster=int(cluster),
                label=lbl,
                growth_z=g if np.isfinite(g) else 0.0,
                inflation_z=i if np.isfinite(i) else 0.0,
                financial_z=f if np.isfinite(f) else 0.0,
                sentiment_z=s_ if np.isfinite(s_) else 0.0,
                vix_z=vix_z if np.isfinite(vix_z) else 0.0,
                n_windows=int(mask.sum()),
            )
        )
    return out


def label_map(labels_list: list[RegimeLabel]) -> dict[int, str]:
    return {rl.cluster: rl.label for rl in labels_list}


def label_from_z(
    growth_z: float = 0.0,
    inflation_z: float = 0.0,
    financial_z: float = 0.0,
    sentiment_z: float = 0.0,
    vix_z: float = float("nan"),
    mode: Mode = "macro",
) -> str:
    """Public entry point: map theme z-scores to a regime label without
    running the full clustering pipeline. Used by sibling projects (e.g.
    accent) that only need a point-in-time regime label.

    `mode="macro"` (default): uses growth x inflation grid with monetary modifier.
    `mode="market"`:          uses monetary x sentiment grid with VIX modifier.
    """
    if mode == "market":
        return _market_base(financial_z, sentiment_z) + _market_suffix(vix_z)
    return _macro_base(growth_z, inflation_z) + _macro_suffix(financial_z)
