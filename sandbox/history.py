from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import sleep

import pandas as pd

from sandbox.mt5 import MT5Adapter, MT5Error
from sandbox.storage import normalize_chunks

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Availability:
    symbol: str
    earliest: str | None
    latest: str | None
    available_bars: int
    missing_intervals: int
    status: str


def chunk_ranges(start: datetime, end: datetime, days: int):
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=days), end)
        yield cursor, stop
        cursor = stop


def download(adapter: MT5Adapter, symbol: str, start: datetime, end: datetime, chunk_days: int, retries: int = 3, progress=None) -> pd.DataFrame:
    chunks = []
    ranges = list(chunk_ranges(start, end, chunk_days))
    for index, (left, right) in enumerate(ranges, 1):
        for attempt in range(1, retries + 1):
            try:
                chunk = adapter.rates(symbol, left, right)
                chunks.append(chunk)
                if progress:
                    progress(index, len(ranges), left, right, len(chunk))
                break
            except MT5Error:
                LOG.exception("historical retrieval failure", extra={"event": "download_retry"})
                if attempt == retries:
                    raise
                sleep(min(2 ** (attempt - 1), 8))
    return normalize_chunks(chunks)


def inspect_history(adapter: MT5Adapter, symbol: str, lookback_years: int = 30, chunk_days: int = 30) -> Availability:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * lookback_years)
    frame = download(adapter, symbol, start, end, chunk_days)
    if frame.empty:
        return Availability(symbol, None, None, 0, 0, "no_data")
    differences = frame.timestamp_utc.diff()
    gaps = int((differences > pd.Timedelta(minutes=1)).sum())
    return Availability(symbol, frame.timestamp_utc.min().isoformat(), frame.timestamp_utc.max().isoformat(), len(frame), gaps, "available")

