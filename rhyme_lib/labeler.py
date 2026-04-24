"""Contextual regime labels from cluster mean theme z-scores.

Primary axes: growth, inflation.
Modifier:    monetary_financial (tight vs. loose / risk-off vs. risk-on).

Label grid:
                      inflation > +.3    neutral |infl| <= .3    inflation < -.3
  growth > +.3    ->  "Reflation"        "Expansion"             "Goldilocks"
  neutral         ->  "Inflationary"     "Neutral"               "Disinflation"
  growth < -.3    ->  "Stagflation"      "Slowdown"              "Deflationary bust"

Appended modifier:
  monetary_financial z > +.5  -> " (risk-off)"
  monetary_financial z < -.5  -> " (risk-on)"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GROWTH = "growth"
INFL = "inflation"
FIN = "monetary_financial"

THRESHOLD = 0.30
RISK_MOD = 0.50


@dataclass
class RegimeLabel:
    cluster: int
    label: str
    growth_z: float
    inflation_z: float
    financial_z: float
    n_windows: int


def _base_label(g: float, i: float) -> str:
    if g > THRESHOLD and i > THRESHOLD:
        return "Reflation"
    if g > THRESHOLD and abs(i) <= THRESHOLD:
        return "Expansion"
    if g > THRESHOLD and i < -THRESHOLD:
        return "Goldilocks"
    if abs(g) <= THRESHOLD and i > THRESHOLD:
        return "Inflationary"
    if abs(g) <= THRESHOLD and abs(i) <= THRESHOLD:
        return "Neutral"
    if abs(g) <= THRESHOLD and i < -THRESHOLD:
        return "Disinflation"
    if g < -THRESHOLD and i > THRESHOLD:
        return "Stagflation"
    if g < -THRESHOLD and abs(i) <= THRESHOLD:
        return "Slowdown"
    return "Deflationary bust"


def _risk_suffix(fin_z: float) -> str:
    if fin_z > RISK_MOD:
        return " (risk-off)"
    if fin_z < -RISK_MOD:
        return " (risk-on)"
    return ""


def label_clusters(
    themes: pd.DataFrame,
    labels: np.ndarray,
    end_dates: pd.DatetimeIndex,
) -> list[RegimeLabel]:
    """For each cluster, compute mean growth/inflation/financial z and assign
    a label. themes is the full theme-aggregated z panel; labels and
    end_dates align window-for-window."""
    aligned = themes.reindex(end_dates)
    missing_axes = [c for c in (GROWTH, INFL, FIN) if c not in aligned.columns]
    if missing_axes:
        raise ValueError(f"themes is missing required columns: {missing_axes}")

    out: list[RegimeLabel] = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        g = float(aligned.loc[mask, GROWTH].mean())
        i = float(aligned.loc[mask, INFL].mean())
        f = float(aligned.loc[mask, FIN].mean())
        base = _base_label(g, i)
        suffix = _risk_suffix(f)
        lbl = "Unclustered" if cluster == -1 else (base + suffix)
        out.append(
            RegimeLabel(
                cluster=int(cluster),
                label=lbl,
                growth_z=g,
                inflation_z=i,
                financial_z=f,
                n_windows=int(mask.sum()),
            )
        )
    return out


def label_map(labels_list: list[RegimeLabel]) -> dict[int, str]:
    return {rl.cluster: rl.label for rl in labels_list}
