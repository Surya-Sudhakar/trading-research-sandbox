from datetime import datetime, timezone
from pathlib import Path

from sandbox.catalog import Catalog
from sandbox.provenance import Provenance


def test_catalog_operations():
    runtime = Path("tests/runtime")
    data = runtime / "catalog-sample.parquet"; data.write_bytes(b"x")
    p = Provenance.create(data, "IC Markets", "server", "EURUSD", "M1", datetime.now(timezone.utc), datetime.now(timezone.utc), 1)
    catalog = Catalog(runtime / "catalog.db")
    catalog.initialize()
    with catalog.connection() as con:
        con.execute("DELETE FROM provenance")
        con.execute("DELETE FROM dataset_provenance_seals")
        con.execute("DELETE FROM datasets")
    catalog.add_dataset(p)
    assert catalog.datasets()[0]["checksum"] == p.checksum
