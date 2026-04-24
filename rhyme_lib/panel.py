"""Panel schema: the default Rhyme panel, its series metadata, and loaders for
the cached parquet that ships in the repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Bucket = Literal["growth", "inflation", "monetary", "sentiment"]
TransformKind = Literal[
    "level",       # stationary: use as-is
    "log_diff",    # log-level with trend: 3m / 12m log-diff
    "bps_diff",    # rate/yield: diff in bps over 3m / 12m
    "yoy_log",     # flow/claims: 4w MA then YoY log-diff
    "already_yoy", # already YoY: use level + 3m diff
]
NativeFreq = Literal["D", "W", "M"]


@dataclass(frozen=True)
class SeriesSpec:
    code: str                  # short column name in the panel
    source_id: str             # FRED id or Yahoo ticker
    source: Literal["fred", "yahoo"]
    bucket: Bucket
    transform: TransformKind
    native_freq: NativeFreq
    description: str


DEFAULT_SPECS: list[SeriesSpec] = [
    # GROWTH
    SeriesSpec("initial_claims",    "ICSA",       "fred",  "growth",             "yoy_log",     "W", "Initial jobless claims"),
    SeriesSpec("continuing_claims", "CCSA",       "fred",  "growth",             "yoy_log",     "W", "Continuing jobless claims"),
    SeriesSpec("ip",                "INDPRO",     "fred",  "growth",             "log_diff",    "M", "Industrial production"),
    SeriesSpec("real_retail_sales", "RRSFS",      "fred",  "growth",             "log_diff",    "M", "Real retail and food services sales"),
    SeriesSpec("wei",               "WEI",        "fred",  "growth",             "level",       "W", "NY Fed Weekly Economic Index"),
    # INFLATION
    SeriesSpec("core_cpi",          "CPILFESL",   "fred",  "inflation",          "log_diff",    "M", "Core CPI"),
    SeriesSpec("core_pce",          "PCEPILFE",   "fred",  "inflation",          "log_diff",    "M", "Core PCE price index"),
    SeriesSpec("trimmed_pce_12m",   "PCETRIM12M159SFRBDAL", "fred", "inflation", "already_yoy", "M", "Dallas Fed trimmed-mean PCE (12m)"),
    SeriesSpec("be5y5y",            "T5YIFR",     "fred",  "inflation",          "level",       "D", "5y5y forward breakeven inflation"),
    SeriesSpec("be10y",             "T10YIE",     "fred",  "inflation",          "level",       "D", "10y breakeven inflation"),
    SeriesSpec("wti",               "DCOILWTICO", "fred",  "inflation",          "log_diff",    "D", "WTI crude oil spot"),
    # MONETARY / FINANCIAL
    SeriesSpec("ff",                "DFF",        "fred",  "monetary", "bps_diff",    "D", "Effective fed funds rate"),
    SeriesSpec("slope_2s10s",       "T10Y2Y",     "fred",  "monetary", "level",       "D", "2s10s Treasury slope"),
    SeriesSpec("slope_3m10y",       "T10Y3M",     "fred",  "monetary", "level",       "D", "3m10y Treasury slope"),
    SeriesSpec("nfci",              "NFCI",       "fred",  "monetary", "level",       "W", "Chicago Fed NFCI financial conditions index"),
    SeriesSpec("vix",               "VIXCLS",     "fred",  "monetary", "level",       "D", "VIX implied volatility"),
    SeriesSpec("baa_spread",        "BAA10YM",    "fred",  "monetary", "level",       "M", "Moody's Baa minus 10y Treasury (credit risk proxy)"),
    SeriesSpec("aaa_spread",        "AAA10YM",    "fred",  "monetary", "level",       "M", "Moody's Aaa minus 10y Treasury"),
    SeriesSpec("dxy",               "DTWEXBGS",   "fred",  "monetary", "log_diff",    "D", "Trade-weighted USD (broad)"),
    SeriesSpec("spx",               "^GSPC",      "yahoo", "monetary", "log_diff",    "D", "S&P 500 index"),
    SeriesSpec("ust10_yield",       "DGS10",      "fred",  "monetary", "bps_diff",    "D", "10y Treasury yield"),
    SeriesSpec("gold",              "GC=F",       "yahoo", "monetary", "log_diff",    "D", "Gold front-month futures (USD/oz)"),
    # SENTIMENT
    SeriesSpec("umich_sentiment",   "UMCSENT",    "fred",  "sentiment",          "level",       "M", "UMich consumer sentiment"),
]

DEFAULT_BUCKETS: dict[str, Bucket] = {s.code: s.bucket for s in DEFAULT_SPECS}
DEFAULT_TRANSFORMS: dict[str, TransformKind] = {s.code: s.transform for s in DEFAULT_SPECS}

PANEL_PATH = Path(__file__).resolve().parent.parent / "data" / "default_panel.parquet"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "default_panel_meta.parquet"


def load_default_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, meta) from the shipped parquet. Raises if missing."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Default panel not found at {PANEL_PATH}. "
            "Run `python refresh_panel.py` to build it (requires FRED_API_KEY)."
        )
    panel = pd.read_parquet(PANEL_PATH)
    meta = pd.read_parquet(META_PATH)
    panel.index = pd.to_datetime(panel.index)
    return panel, meta
