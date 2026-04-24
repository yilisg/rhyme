"""Refresh the cached default panel from FRED, Stooq, Philly Fed, and
Cleveland Fed. Requires FRED_API_KEY in the environment (see .env.example)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rhyme_lib.data_fetch import fetch_default_panel
from rhyme_lib.panel import META_PATH, PANEL_PATH


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print(
            "FRED_API_KEY missing. Copy .env.example to .env and fill it in, "
            "or export FRED_API_KEY in your shell.",
            file=sys.stderr,
        )
        return 2

    print("Fetching default panel...")
    panel, meta = fetch_default_panel(key)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)
    meta.to_parquet(META_PATH)

    print(f"\nPanel  -> {PANEL_PATH}  ({panel.shape[0]:,} rows x {panel.shape[1]} series)")
    print(f"Meta   -> {META_PATH}")
    print("\nSeries status:")
    for _, row in meta.iterrows():
        n = row["n_obs"]
        span = "" if n == 0 else f"{row['start'].date()} -> {row['end'].date()}"
        print(f"  [{row['status']:>6}] {row['code']:<22} {row['bucket']:<20} n={n:<6} {span}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
