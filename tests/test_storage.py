import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sandbox.provenance import Provenance, sha256_file
from sandbox.storage import RawStore
from sandbox.storage import normalize_chunks


def test_chunk_merge_is_sorted_and_deduplicated():
    def chunk(times, opens):
        n = len(times)
        return pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True), "open": opens, "high": [2]*n, "low": [0.5]*n, "close": [1.5]*n, "tick_volume": [1]*n, "spread": [1]*n, "real_volume": [0]*n})
    merged = normalize_chunks([chunk(["2024-01-01 00:01Z", "2024-01-01 00:02Z"], [11, 12]), chunk(["2024-01-01 00:00Z", "2024-01-01 00:01Z"], [10, 99])])
    assert list(merged.open) == [10, 11, 12]
    assert merged.timestamp_utc.is_monotonic_increasing


def test_parquet_round_trip_and_checksum():
    source = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2024-01-01 00:00Z"], utc=True), "open": [1.12345], "high": [1.12355], "low": [1.12340], "close": [1.12350], "tick_volume": [7], "spread": [2], "real_volume": [0]})
    server = "test-" + uuid4().hex
    paths = RawStore(Path("tests/runtime/data")).write_months(source, server, "EURUSD.a", "IC Markets", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
    stored = pd.read_parquet(paths[0])
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), source.reset_index(drop=True))
    provenance = Provenance.create(paths[0], "IC Markets", server, "EURUSD.a", "M1", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc), 1)
    assert provenance.checksum == sha256_file(paths[0])
