from __future__ import annotations
import gzip,json,sqlite3
from pathlib import Path
from uuid import uuid4
import pandas as pd
import pytest
from sandbox.catalog import Catalog
from sandbox.external_import import ExternalImportGateway,ImportSpec,ImportError,PriceType,Resolution
from sandbox.provenance import sha256_file
from sandbox.research.registry import ResearchRegistry

@pytest.fixture
def tmp_path():
    path=(Path("tests/runtime")/("external-import-"+uuid4().hex)).resolve();path.mkdir(parents=True,exist_ok=False);return path

def gateway(tmp_path):return ExternalImportGateway(Catalog(tmp_path/"catalog.sqlite3"),tmp_path/"data")
def m1(path,rows=None):
    pd.DataFrame(rows or [{"timestamp":"2024-01-05 21:59:00","open":1.1,"high":1.2,"low":1.0,"close":1.15},{"timestamp":"2024-01-07 22:01:00","open":1.15,"high":1.25,"low":1.1,"close":1.2}]).to_csv(path,index=False);return path
def mspec(paths,**kw):return ImportSpec(tuple(paths),kw.get("provider","DUKASCOPY"),"EURUSD","EURUSD",Resolution.M1,PriceType.BID,kw.get("timezone","UTC"),{"timestamp":"timestamp","open":"open","high":"high","low":"low","close":"close"},kw.get("broker","IC Markets (EU) Ltd"),False,kw.get("reason"))
def ticks(path,rows=None):
    pd.DataFrame(rows or [{"timestamp":"2024-01-01 00:00:00.000","bid":1.1000,"ask":1.1002},{"timestamp":"2024-01-01 00:00:30.000","bid":1.1010,"ask":1.1013},{"timestamp":"2024-01-01 00:01:00.000","bid":1.0990,"ask":1.0992}]).to_csv(path,index=False);return path
def tspec(paths,mid=False,**kw):return ImportSpec(tuple(paths),kw.get("provider","DUKASCOPY"),"EURUSD","EURUSD",Resolution.TICK,PriceType.BID_ASK,kw.get("timezone","UTC"),{"timestamp":"timestamp","bid":"bid","ask":"ask"},kw.get("broker","IC Markets (EU) Ltd"),mid,kw.get("reason"))

def test_valid_csv_preserves_source_provenance_certifies_and_resolves(tmp_path):
    source=m1(tmp_path/"ICMarkets_EURUSD.csv");original=sha256_file(source);g=gateway(tmp_path);r=g.import_files(mspec([source]));assert r["status"]=="STAGED"
    info=g.inspect_dataset(r["dataset_id"]);m=info["manifest"];assert m["data_source_provider"]=="DUKASCOPY" and m["target_execution_broker"]=="IC Markets (EU) Ltd" and m["provider_authentication"]=="DECLARED_PROVENANCE";assert m["component_sha256s"]==[original]
    assert g.certify(r["dataset_id"])["classification"]=="UNASSIGNED_RESEARCH_DATA";assert ResearchRegistry(tmp_path/"catalog.sqlite3",tmp_path,tmp_path/"results").verify_dataset(r["dataset_id"])["checksum"]==m["canonical_dataset_sha256"]
    with g.catalog.connection() as con:assert con.execute("select count(*) from sqlite_master where type='table' and name='data_partitions'").fetchone()[0]==0

def test_parquet_and_csv_gz_supported(tmp_path):
    frame=pd.DataFrame([{"timestamp":"2024-01-01","open":1,"high":2,"low":.5,"close":1.5}]);p=tmp_path/"a.parquet";frame.to_parquet(p,index=False);assert gateway(tmp_path/"p").import_files(mspec([p]))["status"]=="STAGED"
    z=tmp_path/"a.csv.gz";frame.to_csv(z,index=False,compression="gzip");assert gateway(tmp_path/"z").import_files(mspec([z]))["status"]=="STAGED"

def test_duplicate_and_multiple_interpretation_are_explicit(tmp_path):
    p=m1(tmp_path/"a.csv");g=gateway(tmp_path);first=g.import_files(mspec([p]));dup=g.import_files(mspec([p]));assert dup["outcome"]=="DUPLICATE_DATASET" and dup["existing_dataset_id"]==first["dataset_id"]
    with pytest.raises(ImportError,match="MULTIPLE_INTERPRETATIONS"):g.import_files(mspec([p],timezone="Europe/Berlin"))
    alt=g.import_files(mspec([p],timezone="Europe/Berlin",reason="Source documentation is disputed"));assert alt["outcome"]=="MULTIPLE_INTERPRETATIONS" and alt["dataset_id"]!=first["dataset_id"]

def test_timezone_is_required_and_ambiguous_local_time_quarantines(tmp_path):
    p=m1(tmp_path/"a.csv");g=gateway(tmp_path)
    with pytest.raises(ImportError,match="timezone"):g.import_files(mspec([p],timezone="UNKNOWN"))
    m1(p,[{"timestamp":"2024-10-27 02:30:00","open":1,"high":2,"low":.5,"close":1.5}])
    with pytest.raises(ImportError,match="no valid rows"):g.import_files(mspec([p],timezone="Europe/Berlin"))

@pytest.mark.parametrize("value",[float("nan"),float("inf"),-1.0])
def test_invalid_numeric_prices_quarantine_and_cannot_certify(tmp_path,value):
    p=m1(tmp_path/"bad.csv",[{"timestamp":"2024-01-01","open":value,"high":2,"low":.5,"close":1.5}]);g=gateway(tmp_path);r=g.import_files(mspec([p]));assert r["status"]=="QUARANTINED"
    with pytest.raises(ImportError,match="cannot certify"):g.certify(r["dataset_id"])

def test_invalid_ohlc_duplicate_and_out_of_order_quarantine(tmp_path):
    rows=[{"timestamp":"2024-01-01 00:01","open":1,"high":.9,"low":.8,"close":1.1},{"timestamp":"2024-01-01 00:00","open":1,"high":2,"low":.5,"close":1.5},{"timestamp":"2024-01-01 00:00","open":1,"high":2,"low":.5,"close":1.5}]
    g=gateway(tmp_path);r=g.import_files(mspec([m1(tmp_path/"bad.csv",rows)]));assert r["status"]=="QUARANTINED";a=r["audit"];assert a["ohlc_violations"]==1 and a["duplicate_events"]>=1 and a["ordering_issues"]>=1

def test_gap_classification_does_not_fill_minutes(tmp_path):
    g=gateway(tmp_path);r=g.import_files(mspec([m1(tmp_path/"weekend.csv")])) ;assert r["audit"]["expected_gaps"]==1
    rows=[{"timestamp":"2024-01-01 00:00","open":1,"high":2,"low":.5,"close":1.5},{"timestamp":"2024-01-01 00:10","open":1,"high":2,"low":.5,"close":1.5}];r2=g.import_files(mspec([m1(tmp_path/"gap.csv",rows)]));assert r2["audit"]["suspicious_gaps"]==1 and g.inspect_dataset(r2["dataset_id"])["manifest"]["row_count"]==2

def test_ticks_bid_ask_mid_spreads_empty_minute_and_half_open_boundary(tmp_path):
    p=ticks(tmp_path/"ticks.csv");g=gateway(tmp_path);r=g.import_files(tspec([p],mid=True));assert r["status"]=="STAGED"
    bid=g.derive_m1(r["dataset_id"],PriceType.BID);ask=g.derive_m1(r["dataset_id"],PriceType.ASK);mid=g.derive_m1(r["dataset_id"],PriceType.MID)
    assert len(bid)==2 and bid.iloc[0][["open","high","low","close"]].tolist()==[1.1,1.101,1.1,1.101]
    assert ask.iloc[0][["open","high","low","close"]].tolist()==[1.1002,1.1013,1.1002,1.1013];assert mid.iloc[0].open==pytest.approx(1.1001);assert bid.iloc[0].spread_max==pytest.approx(.0003)
    r2=gateway(tmp_path/"nomid").import_files(tspec([p],mid=False))
    with pytest.raises(ImportError,match="explicitly requested"):gateway(tmp_path/"nomid").derive_m1(r2["dataset_id"],PriceType.MID)

def test_crossed_quote_quarantined_but_same_timestamp_distinct_ticks_allowed(tmp_path):
    rows=[{"timestamp":"2024-01-01","bid":1.1,"ask":1.2},{"timestamp":"2024-01-01","bid":1.2,"ask":1.3},{"timestamp":"2024-01-01 00:01","bid":1.4,"ask":1.3}];g=gateway(tmp_path);r=g.import_files(tspec([ticks(tmp_path/"t.csv",rows)]));assert r["status"]=="QUARANTINED" and r["audit"]["crossed_quotes"]==1 and r["audit"]["duplicate_events"]==0

def test_multifile_order_deterministic_and_overlap_detected(tmp_path):
    a=m1(tmp_path/"a.csv",[{"timestamp":"2024-01-01","open":1,"high":2,"low":.5,"close":1.5}]);b=m1(tmp_path/"b.csv",[{"timestamp":"2024-01-02","open":1,"high":2,"low":.5,"close":1.5}]);g=gateway(tmp_path);r=g.import_files(mspec([b,a]));assert g.inspect_dataset(r["dataset_id"])["manifest"]["original_filenames"]==["a.csv","b.csv"]
    assert g.import_files(mspec([a,b]))["outcome"]=="DUPLICATE_DATASET"
    c=m1(tmp_path/"c.csv",[{"timestamp":"2024-01-01","open":1,"high":2,"low":.5,"close":1.5}]);r2=g.import_files(mspec([a,c]));assert r2["status"]=="QUARANTINED" and r2["audit"]["duplicate_events"]>=1

def test_source_change_blocks_certification_and_no_false_partition(tmp_path):
    p=m1(tmp_path/"a.csv");g=gateway(tmp_path);r=g.import_files(mspec([p]));p.write_text("changed",encoding="utf-8")
    with pytest.raises(ImportError,match="checksum mismatch"):g.certify(r["dataset_id"])
    assert not g.catalog.datasets()

def test_path_metadata_rejected_and_no_partial_catalog(tmp_path):
    p=m1(tmp_path/"a.csv");g=gateway(tmp_path)
    with pytest.raises(ImportError,match="unsafe provider"):g.import_files(mspec([p],provider="../../escape"))
    assert g.list_imports()==[]

def test_catalog_registration_failure_removes_promoted_artifacts(tmp_path):
    p=m1(tmp_path/"a.csv");g=gateway(tmp_path)
    with g.catalog.connection() as con:con.execute("CREATE TRIGGER reject_external BEFORE INSERT ON external_imports BEGIN SELECT RAISE(ABORT,'fixture failure'); END")
    with pytest.raises(sqlite3.IntegrityError,match="fixture failure"):g.import_files(mspec([p]))
    assert not list((tmp_path/"data"/"external_raw").rglob("*.csv")) and not list((tmp_path/"data"/"derived"/"external").rglob("*.parquet")) and g.list_imports()==[]

def test_manifest_tampering_is_detected(tmp_path):
    g=gateway(tmp_path);r=g.import_files(mspec([m1(tmp_path/"a.csv")]))
    with g.catalog.connection() as con:
        value=json.loads(con.execute("select manifest_json from external_imports where dataset_id=?",(r["dataset_id"],)).fetchone()[0]);value["data_source_provider"]="IC Markets";con.execute("update external_imports set manifest_json=? where dataset_id=?",(json.dumps(value,sort_keys=True),r["dataset_id"]))
    with pytest.raises(ImportError,match="manifest fingerprint"):g.inspect_dataset(r["dataset_id"])

def test_chunked_writer_handles_multiple_batches(tmp_path,monkeypatch):
    import sandbox.external_import as module
    p=m1(tmp_path/"a.csv");original=module._chunks
    def split(path,columns,size=200_000):
        frame=pd.read_csv(path,usecols=columns);yield frame.iloc[:1];yield frame.iloc[1:]
    monkeypatch.setattr(module,"_chunks",split);g=gateway(tmp_path);r=g.import_files(mspec([p]));assert g.inspect_dataset(r["dataset_id"])["manifest"]["row_count"]==2
