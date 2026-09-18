import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import pytest

from sandbox.catalog import Catalog
from sandbox.provenance import Provenance, sha256_file
from sandbox.storage import ObservationConflictError, RawStore
from sandbox.storage import normalize_chunks


def bars(times, close=1.15):
    count=len(times)
    return pd.DataFrame({"timestamp_utc":pd.to_datetime(times,utc=True),"open":[1.1]*count,
        "high":[1.2]*count,"low":[1.0]*count,"close":[close]*count,
        "tick_volume":[1]*count,"spread":[1]*count,"real_volume":[0]*count})


def catalog_store():
    root=Path("tests/runtime")/("raw-store-"+uuid4().hex)
    catalog=Catalog(root/"catalog.sqlite3")
    return root,catalog,RawStore(root/"data",catalog)


def test_chunk_merge_is_sorted_and_deduplicated():
    def chunk(times, opens):
        n = len(times)
        return pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True), "open": opens, "high": [2]*n, "low": [0.5]*n, "close": [1.5]*n, "tick_volume": [1]*n, "spread": [1]*n, "real_volume": [0]*n})
    merged = normalize_chunks([chunk(["2024-01-01 00:01Z", "2024-01-01 00:02Z"], [11, 12]), chunk(["2024-01-01 00:00Z", "2024-01-01 00:01Z"], [10, 11])])
    assert list(merged.open) == [10, 11, 12]
    assert merged.timestamp_utc.is_monotonic_increasing


def test_chunk_merge_rejects_conflicting_overlap():
    left=bars(["2024-01-01 00:00Z"])
    right=bars(["2024-01-01 00:00Z"],close=1.16)
    with pytest.raises(ObservationConflictError,match="conflicting observations"):
        normalize_chunks([left,right])


def test_parquet_round_trip_and_checksum():
    source = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2024-01-01 00:00Z"], utc=True), "open": [1.12345], "high": [1.12355], "low": [1.12340], "close": [1.12350], "tick_volume": [7], "spread": [2], "real_volume": [0]})
    server = "test-" + uuid4().hex
    paths = RawStore(Path("tests/runtime/data")).write_months(source, server, "EURUSD.a", "IC Markets", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
    stored = pd.read_parquet(paths[0])
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), source.reset_index(drop=True))
    provenance = Provenance.create(paths[0], "IC Markets", server, "EURUSD.a", "M1", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc), 1)
    assert provenance.checksum == sha256_file(paths[0])


def test_missing_bar_is_preserved_with_integrity_evidence():
    source = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 00:02Z"], utc=True), "open": [1.1, 1.1], "high": [1.2, 1.2], "low": [1.0, 1.0], "close": [1.15, 1.15], "tick_volume": [1, 1], "spread": [1, 1], "real_volume": [0, 0]})
    server = "test-gap-" + uuid4().hex
    path = RawStore(Path("tests/runtime/data")).write_months(source, server, "EURUSD", "IC Markets", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))[0]
    pd.testing.assert_frame_equal(pd.read_parquet(path).reset_index(drop=True), source)
    report = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))["integrity_report"]
    assert report["status"] == "warning"
    assert report["issues"][0]["code"] == "missing_interval"


def test_partial_month_merge_and_exact_retry_are_idempotent():
    root,catalog,store=catalog_store();start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    path=store.write_months(bars(["2024-01-01 00:00Z","2024-01-01 00:01Z"]),"SERVER","EURUSD","BROKER",start,end)[0]
    dataset_id=catalog.dataset_for_path(path)["dataset_id"]
    expanded=bars(["2024-01-01 00:00Z","2024-01-01 00:01Z","2024-01-01 00:02Z"])
    store.write_months(expanded,"SERVER","EURUSD","BROKER",start,end)
    assert len(pd.read_parquet(path))==3
    assert catalog.dataset_for_path(path)["dataset_id"]==dataset_id
    before=(path.read_bytes(),path.with_suffix(".metadata.json").read_bytes())
    store.write_months(expanded,"SERVER","EURUSD","BROKER",start,end)
    assert before==(path.read_bytes(),path.with_suffix(".metadata.json").read_bytes())
    with catalog.connection() as con:
        assert con.execute("SELECT count(*) FROM datasets WHERE path=?",(str(path),)).fetchone()[0]==1
        assert con.execute("SELECT count(*) FROM raw_dataset_revisions WHERE dataset_id=?",(dataset_id,)).fetchone()[0]==2
        current=con.execute("SELECT checksum FROM datasets WHERE dataset_id=?",(dataset_id,)).fetchone()[0]
    assert current==sha256_file(path)


def test_conflicting_existing_overlap_preserves_evidence_and_target():
    root,catalog,store=catalog_store();start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    path=store.write_months(bars(["2024-01-01 00:00Z"]),"SERVER","EURUSD","BROKER",start,end)[0]
    original=path.read_bytes()
    with pytest.raises(ObservationConflictError,match="evidence preserved") as caught:
        store.write_months(bars(["2024-01-01 00:00Z"],close=1.16),"SERVER","EURUSD","BROKER",start,end)
    assert path.read_bytes()==original and caught.value.evidence_path.exists()
    evidence=pd.read_parquet(caught.value.evidence_path)
    assert set(evidence.observation_source)=={"existing","incoming"} and len(evidence)==2


def test_multimonth_retry_adopts_orphan_and_completes_catalog():
    root,catalog,_=catalog_store();start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,3,1,tzinfo=timezone.utc)
    orphan_store=RawStore(root/"data")
    january=bars(["2024-01-31 23:59Z"])
    january_path=orphan_store.write_months(january,"SERVER","EURUSD","BROKER",start,end)[0]
    assert catalog.dataset_for_path(january_path) is None
    store=RawStore(root/"data",catalog)
    paths=store.write_months(pd.concat([january,bars(["2024-02-01 00:00Z"])]),"SERVER","EURUSD","BROKER",start,end)
    assert [path.name for path in paths]==["01.parquet","02.parquet"]
    assert len(catalog.datasets())==2 and all(catalog.dataset_for_path(path) for path in paths)


def test_continuity_v2_metadata_is_recomputed_after_merge():
    root,catalog,store=catalog_store();start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    store.write_months(bars(["2024-01-01 00:00Z"]),"SERVER","EURUSD","BROKER",start,end)
    path=store.write_months(bars(["2024-01-01 00:00Z","2024-01-01 00:02Z"]),"SERVER","EURUSD","BROKER",start,end)[0]
    metadata=json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["continuity_policy"]["version"]=="CONTINUITY_V2"
    assert metadata["continuity_segment_count"]==2
    assert metadata["gaps"][0]["missing_slots"]==1 and metadata["gaps"][0]["reset"]


def test_catalog_failure_after_atomic_write_recovers_without_losing_revision(monkeypatch):
    root,catalog,store=catalog_store()
    start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    original=bars(["2024-01-01 00:00Z"])
    path=store.write_months(original,"SERVER","EURUSD","BROKER",start,end)[0]
    old_bytes=path.read_bytes();old_checksum=sha256_file(path)
    expanded=bars(["2024-01-01 00:00Z","2024-01-01 00:02Z"])
    update=catalog.upsert_raw_dataset
    def fail(_):
        raise RuntimeError("injected catalog failure")
    monkeypatch.setattr(catalog,"upsert_raw_dataset",fail)
    with pytest.raises(RuntimeError,match="injected"):
        store.write_months(expanded,"SERVER","EURUSD","BROKER",start,end)
    assert path.with_suffix(".pending.json").exists()
    assert (path.parent/".revisions"/path.stem/(old_checksum+".parquet")).read_bytes()==old_bytes
    monkeypatch.setattr(catalog,"upsert_raw_dataset",update)
    store.write_months(expanded,"SERVER","EURUSD","BROKER",start,end)
    assert not path.with_suffix(".pending.json").exists()
    assert catalog.dataset_for_path(path)["checksum"]==sha256_file(path)
    pd.testing.assert_frame_equal(pd.read_parquet(path),expanded)


def test_path_alias_retry_reuses_catalog_identity():
    root,catalog,store=catalog_store()
    start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    frame=bars(["2024-01-01 00:00Z"])
    path=store.write_months(frame,"SERVER","EURUSD","BROKER",start,end)[0]
    identity=catalog.dataset_for_path(path)["dataset_id"]
    RawStore((root/"data").resolve(),catalog).write_months(frame,"SERVER","EURUSD","BROKER",start,end)
    assert len(catalog.datasets())==1
    assert catalog.dataset_for_path(path.resolve())["dataset_id"]==identity


def test_tampered_month_is_not_silently_resealed():
    root,catalog,store=catalog_store()
    start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    frame=bars(["2024-01-01 00:00Z"])
    path=store.write_months(frame,"SERVER","EURUSD","BROKER",start,end)[0]
    registered=catalog.dataset_for_path(path)
    bars(["2024-01-01 00:00Z"],close=1.16).to_parquet(path,index=False)
    with pytest.raises(ValueError,match="checksum mismatch"):
        store.write_months(frame,"SERVER","EURUSD","BROKER",start,end)
    assert catalog.dataset_for_path(path)==registered


def test_corrupt_provenance_is_not_resealed():
    root,catalog,store=catalog_store()
    start=datetime(2024,1,1,tzinfo=timezone.utc);end=datetime(2024,2,1,tzinfo=timezone.utc)
    frame=bars(["2024-01-01 00:00Z"])
    path=store.write_months(frame,"SERVER","EURUSD","BROKER",start,end)[0]
    with catalog.connection() as con:
        con.execute("UPDATE provenance SET record_json='{}'")
    with pytest.raises(ValueError,match="seal mismatch"):
        store.write_months(frame,"SERVER","EURUSD","BROKER",start,end)
