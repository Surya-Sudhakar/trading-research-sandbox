from datetime import datetime, timezone
from pathlib import Path

from sandbox.provenance import Provenance, sha256_file


def test_checksum_and_provenance_are_reproducible():
    path = Path("tests/runtime/provenance-sample.parquet"); path.write_bytes(b"immutable-data")
    assert sha256_file(path) == sha256_file(path)
    p = Provenance.create(path, "IC Markets", "demo", "EURUSD", "M1", datetime.now(timezone.utc), datetime.now(timezone.utc), 3)
    assert p.checksum == sha256_file(path)
    assert p.dataset_id and p.row_count == 3
