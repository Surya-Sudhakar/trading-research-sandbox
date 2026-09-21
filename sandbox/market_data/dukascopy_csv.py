"""Bounded-memory pre-2025 dukascopy-node CSV validation and BID M15 aggregation."""
from pathlib import Path
import numpy as np
import pandas as pd

LIMIT_MS = 1735689600000  # 2025-01-01 UTC, exclusive
FLOOR_MS = 1451606400000  # 2016-01-01 UTC, inclusive
BUCKET_MS = 900000


def aggregate_csv(path, chunksize=250_000):
    """Preserve source order and duplicates; raw Bid/Ask remain immutable in source."""
    path = Path(path)
    parts, monthly = [], {}
    previous = first = last = None
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunksize, keep_default_na=False):
        if list(chunk.columns) != ['timestamp', 'askPrice', 'bidPrice']:
            raise ValueError('expected dukascopy-node timestamp,askPrice,bidPrice schema')
        if not chunk.timestamp.str.fullmatch(r'[0-9]+').all():
            raise ValueError('integer epoch milliseconds required')
        t = chunk.timestamp.to_numpy(dtype=np.int64)
        # Check timestamps before interpreting any quote values.
        if (t < FLOOR_MS).any() or (t >= LIMIT_MS).any():
            raise ValueError('source outside authorized 2016-2024 window')
        delta = np.diff(t, prepend=previous if previous is not None else t[0]-1)
        if (delta < 0).any(): raise ValueError('backward timestamps')
        bid = chunk.bidPrice.to_numpy(float); ask = chunk.askPrice.to_numpy(float)
        if not np.isfinite(bid).all() or not np.isfinite(ask).all() or (bid <= 0).any() or (ask <= 0).any():
            raise ValueError('invalid Bid/Ask quote')
        if (ask < bid).any(): raise ValueError(f'Ask below Bid: {int((ask < bid).sum())} rows in chunk; source preserved')
        times = pd.to_datetime(t, unit='ms', utc=True)
        month_codes = t.astype('datetime64[ms]').astype('datetime64[M]').astype(np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(month_codes))+1]
        ends = np.r_[starts[1:], len(t)]
        for begin, end in zip(starts, ends):
            month = times[begin].strftime('%Y-%m')
            record = monthly.setdefault(month, {'raw_tick_count': 0, 'duplicate_timestamps': 0,
                'first_timestamp': times[begin].isoformat(), 'last_timestamp': None})
            record['raw_tick_count'] += int(end-begin)
            record['duplicate_timestamps'] += int((delta[begin:end] == 0).sum())
            record['last_timestamp'] = times[end-1].isoformat()
        # Grouping never sorts/reorders ticks within a bucket. Cross-chunk buckets merge below.
        frame = pd.DataFrame({'bucket': t//BUCKET_MS, 'bid': bid})
        part = frame.groupby('bucket', sort=False).bid.agg(['first', 'max', 'min', 'last', 'count'])
        parts.append(part)
        if first is None: first = int(t[0])
        last = previous = int(t[-1])
    if first is None: raise ValueError('empty tick file')
    combined = pd.concat(parts)
    candles = combined.groupby(level=0, sort=False).agg({'first': 'first', 'max': 'max', 'min': 'min', 'last': 'last', 'count': 'sum'})
    candles = candles.rename(columns={'first': 'open', 'max': 'high', 'min': 'low', 'last': 'close', 'count': 'tick_count'})
    candles.insert(0, 'timestamp_utc', pd.to_datetime(candles.index.to_numpy()*BUCKET_MS, unit='ms', utc=True))
    candles = candles.reset_index(drop=True)
    # Same conservative boundary rule as cross_feed_diagnostic.aggregate.
    lower = pd.Timestamp(first, unit='ms', tz='UTC').ceil('15min')
    upper = pd.Timestamp(last, unit='ms', tz='UTC').floor('15min')
    candles['partial_boundary'] = (candles.timestamp_utc < lower) | (candles.timestamp_utc >= upper)
    return candles, monthly
