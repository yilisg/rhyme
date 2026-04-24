"""Fetch the default Rhyme panel from FRED and Yahoo Finance.
Called by refresh_panel.py."""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf
from fredapi import Fred

from .panel import DEFAULT_SPECS, SeriesSpec

log = logging.getLogger("rhyme.fetch")

FRED_START = "1950-01-01"
YAHOO_START = "1950-01-01"
MAX_FRED_RETRIES = 3
FRED_RETRY_BACKOFF = 1.5


def fetch_fred(fred: Fred, series_id: str) -> pd.Series:
    last_err: Optional[Exception] = None
    for attempt in range(MAX_FRED_RETRIES):
        try:
            s = fred.get_series(series_id, observation_start=FRED_START)
            s.index = pd.to_datetime(s.index)
            return s.astype(float).rename(series_id)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "Internal Server Error" in msg or "Bad Gateway" in msg or "Timeout" in msg:
                time.sleep(FRED_RETRY_BACKOFF ** attempt)
                continue
            raise
    assert last_err is not None
    raise last_err


def fetch_yahoo(ticker: str) -> pd.Series:
    df = yf.download(
        ticker,
        start=YAHOO_START,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned empty frame for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise RuntimeError(f"Yahoo CSV missing Close for {ticker}: {df.columns.tolist()}")
    s = df["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
    return s.rename(ticker)


def fetch_one(spec: SeriesSpec, fred: Fred) -> Optional[pd.Series]:
    try:
        if spec.source == "fred":
            s = fetch_fred(fred, spec.source_id)
        elif spec.source == "yahoo":
            s = fetch_yahoo(spec.source_id)
        else:
            raise ValueError(f"unknown source {spec.source}")
        return s.rename(spec.code)
    except Exception as e:
        log.warning("fetch failed for %s (%s): %s", spec.code, spec.source_id, e)
        return None


def fetch_default_panel(fred_api_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch every series in DEFAULT_SPECS, align on a daily index, return
    (panel_df, meta_df). Panel is daily with forward-fill (90-day limit) so
    downstream code can resample cleanly to weekly or monthly."""
    fred = Fred(api_key=fred_api_key)
    series_list: list[pd.Series] = []
    meta_rows: list[dict] = []
    for spec in DEFAULT_SPECS:
        log.info("fetching %s from %s (%s)", spec.code, spec.source, spec.source_id)
        s = fetch_one(spec, fred)
        status = "ok" if s is not None and len(s) > 0 else "failed"
        if s is not None and len(s) > 0:
            series_list.append(s)
        meta_rows.append(
            {
                "code": spec.code,
                "source_id": spec.source_id,
                "source": spec.source,
                "bucket": spec.bucket,
                "transform": spec.transform,
                "native_freq": spec.native_freq,
                "description": spec.description,
                "status": status,
                "n_obs": 0 if s is None else int(len(s)),
                "start": None if s is None or len(s) == 0 else s.index.min(),
                "end":   None if s is None or len(s) == 0 else s.index.max(),
            }
        )

    if not series_list:
        raise RuntimeError("no series fetched successfully")

    idx = pd.DatetimeIndex(
        pd.date_range(
            min(s.index.min() for s in series_list),
            max(s.index.max() for s in series_list),
            freq="D",
        )
    )
    panel = pd.DataFrame(index=idx)
    for s in series_list:
        panel[s.name] = s.reindex(idx).ffill(limit=90)
    meta = pd.DataFrame(meta_rows)
    return panel, meta
