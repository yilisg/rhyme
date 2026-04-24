"""Compute forward returns (or forward changes) for the assets the analog
table cares about: SPX, UST10y yield, DXY, Baa spread, Aaa spread, Gold, WTI.

For price assets: forward log return. For yields/spreads: forward bps change.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_ASSETS: dict[str, str] = {
    "spx": "ret",
    "ust10_yield": "bps",
    "dxy": "ret",
    "baa_spread": "bps",
    "aaa_spread": "bps",
    "gold": "ret",
    "wti": "ret",
}
DEFAULT_HORIZONS_WEEKS = {"1m": 4, "3m": 13, "12m": 52}


def forward_returns(
    panel_daily: pd.DataFrame,
    ref_dates: pd.DatetimeIndex,
    assets: Mapping[str, str] | None = None,
    horizon_weeks: int = 13,
) -> pd.DataFrame:
    assets = assets or DEFAULT_ASSETS
    days = horizon_weeks * 7

    out = pd.DataFrame(index=ref_dates)
    for asset, kind in assets.items():
        if asset not in panel_daily.columns:
            out[asset] = np.nan
            continue
        col = panel_daily[asset].ffill(limit=30)
        fwd = col.reindex(ref_dates + pd.Timedelta(days=days), method="ffill")
        fwd.index = ref_dates
        now = col.reindex(ref_dates, method="ffill")
        if kind == "ret":
            out[asset] = np.log(fwd / now)
        else:  # bps change
            out[asset] = fwd - now
    return out
