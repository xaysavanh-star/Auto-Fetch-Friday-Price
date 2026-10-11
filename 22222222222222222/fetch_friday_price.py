#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch a public gold futures quote and append it to the Friday price log."""

import csv
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf


EXPECTED_COLUMNS = ["week_of", "fetch_time_utc", "leg", "price", "source"]
LEG = os.environ.get("FRIDAY_LEG", "").strip().lower()
TICKER = "GC=F"
SOURCE = f"Yahoo Finance {TICKER}"


def main():
    if LEG not in {"entry", "exit"}:
        print("FRIDAY_LEG must be either 'entry' or 'exit'.", file=sys.stderr)
        return 2

    data = yf.Ticker(TICKER).history(period="1d", interval="1m")
    if data.empty or "Close" not in data:
        print(f"No recent price data returned for {TICKER}.", file=sys.stderr)
        return 1

    closes = data["Close"].dropna()
    if closes.empty:
        print(f"No valid closing prices returned for {TICKER}.", file=sys.stderr)
        return 1

    price = float(closes.iloc[-1])
    if not math.isfinite(price) or price <= 0:
        print(f"Invalid price returned for {TICKER}: {price!r}.", file=sys.stderr)
        return 1

    fetched_at = datetime.now(timezone.utc)
    week_of = (fetched_at.date() - timedelta(days=fetched_at.weekday())).isoformat()
    row = {
        "week_of": week_of,
        "fetch_time_utc": fetched_at.isoformat(timespec="seconds"),
        "leg": LEG,
        "price": f"{price:.6f}",
        "source": SOURCE,
    }

    log_file = Path(__file__).with_name("friday_auto_price_log.csv")
    write_header = not log_file.exists() or log_file.stat().st_size == 0
    if not write_header:
        with log_file.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != EXPECTED_COLUMNS:
            print(
                f"Unexpected CSV header in {log_file}: {existing_header!r}",
                file=sys.stderr,
            )
            return 1

    with log_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"Logged {LEG} price for week {week_of}: "
        f"{price:.6f} at {row['fetch_time_utc']} UTC"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
