from __future__ import annotations
import gzip,io,json,sqlite3
from pathlib import Path
from uuid import uuid4
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sandbox.api.app import UPLOAD_CHUNK_BYTES,create_app
from sandbox.api.service import SandboxReadService
from sandbox.config import Settings

@pytest.fixture
def root():
    value=(Path("tests/runtime")/("import-api-"+uuid4().hex)).resolve();value.mkdir(parents=True);return value

@pytest.fixture
def client(root):
    settings=Settings(None,root/"data",root/"catalog.sqlite3",7,3,"INFO");return TestClient(create_app(SandboxReadService(settings,root,probe_broker=False)))

def csv_bytes(rows=None):
    frame=pd.DataFrame(rows or [{"timestamp":"2024-01-01 00:00","open":1.1,"high":1.2,"low":1.0,"close":1.15},{"timestamp":"2024-01-01 00:01","open":1.15,"high":1.25,"low":1.1,"close":1.2}]);return frame.to_csv(index=False).encode()

def fields(**values):
    base={"provider":"DUKASCOPY","symbol":"EURUSD","data_type":"M1","price_type":"BID","source_timezone":"UTC","purpose":"UNASSIGNED_RESEARCH_DATA"};base.update(values);return base

def upload(client,name="history.csv",body=None,**metadata):return client.post("/api/data/import",data=fields(**metadata),files={"file":(name,body or csv_bytes(),"application/octet-stream")})

def test_valid_csv_upload_certifies_catalogs_and_exposes_no_path(client):
    response=upload(client);assert response.status_code==200;x=response.json();assert x["certification"]=="CERTIFIED" and x["row_count"]==2 and len(x["original_sha256"])==64 and x["provenance"]=="DECLARED_PROVENANCE"
    catalog=client.get("/api/data/status");assert catalog.json()["imported_datasets"][0]["dataset_id"]==x["dataset_id"] and '"path"' not in catalog.text.lower() and "tests\\runtime" not in catalog.text.lower()

def test_valid_csv_gz_and_parquet_uploads(client):
    zipped=gzip.compress(csv_bytes());assert upload(client,"history.csv.gz",zipped).json()["certification"]=="CERTIFIED"
    buffer=io.BytesIO();pd.read_csv(io.BytesIO(csv_bytes())).to_parquet(buffer,index=False);assert upload(client,"history.parquet",buffer.getvalue()).json()["certification"]=="CERTIFIED"

def test_unsupported_extension_and_path_traversal_rejected(client):
    assert upload(client,"history.exe").status_code==400
    assert upload(client,"../history.csv").status_code==400
    assert upload(client,"..\\history.csv").status_code==400

def test_duplicate_upload_is_explicit_and_does_not_duplicate_catalog(client):
    first=upload(client).json();second=upload(client).json();assert second["certification"]=="DUPLICATE" and second["dataset_id"]==first["dataset_id"]
    assert len(client.get("/api/data/status").json()["imported_datasets"])==1

def test_invalid_schema_and_timestamp_are_safe_rejections(client):
    bad_schema=upload(client,body=b"when,value\n2024-01-01,1\n");assert bad_schema.status_code==200 and bad_schema.json()["certification"]=="REJECTED" and bad_schema.json()["validation_summary"]["schema_error_count"]==1 and "path" not in bad_schema.text.lower()
    bad_time=upload(client,body=csv_bytes([{"timestamp":"not-a-date","open":1,"high":2,"low":.5,"close":1.5}]));assert bad_time.status_code==200 and bad_time.json()["validation_summary"]["invalid_timestamp_count"]==1 and "Traceback" not in bad_time.text

def test_crossed_quotes_and_invalid_ohlc_are_quarantined_without_bypass(client):
    ticks=pd.DataFrame([{"timestamp":"2024-01-01","bid":1.2,"ask":1.1}]).to_csv(index=False).encode();crossed=upload(client,"ticks.csv",ticks,data_type="TICK",price_type="BID_ASK").json();assert crossed["certification"]=="QUARANTINED" and crossed["crossed_quotes"]==1 and "NOT ELIGIBLE" in crossed["message"]
    ohlc=csv_bytes([{"timestamp":"2024-01-01","open":1,"high":.9,"low":.8,"close":1.1}]);invalid=upload(client,"invalid.csv",ohlc).json();assert invalid["certification"]=="QUARANTINED" and invalid["invalid_rows"]==1

@pytest.mark.parametrize("overrides",[{"provider":"UNKNOWN_VENDOR"},{"purpose":"VALIDATION"},{"purpose":"FINAL_TEST"},{"data_type":"H1"},{"price_type":"MAGIC"},{"source_timezone":""},{"symbol":"../../EURUSD"}])
def test_metadata_validation_blocks_unsafe_or_stage6_values(client,overrides):assert upload(client,**overrides).status_code in (400,422)

def test_configured_size_limit_and_streaming_chunk_constant(client,monkeypatch):
    monkeypatch.setenv("SANDBOX_UPLOAD_MAX_BYTES","1024");assert UPLOAD_CHUNK_BYTES==1024*1024;response=upload(client,body=b"x"*2048);assert response.status_code==413

def test_engineering_diagnostic_uses_existing_classification_without_partition(client,root):
    x=upload(client,purpose="ENGINEERING_DIAGNOSTIC").json();assert x["certification"]=="CERTIFIED"
    con=sqlite3.connect(root/"catalog.sqlite3");partitions=con.execute("select count(*) from data_partitions").fetchone()[0] if con.execute("select 1 from sqlite_master where type='table' and name='data_partitions'").fetchone() else 0;assert con.execute("select count(*) from dataset_classifications where dataset_id=? and classification='ENGINEERING_DIAGNOSTIC'",(x["dataset_id"],)).fetchone()[0]==1 and partitions==0

def test_import_does_not_create_research_or_execution_state(client,root):
    upload(client);con=sqlite3.connect(root/"catalog.sqlite3")
    def count(table):return con.execute("select count(*) from "+table).fetchone()[0] if con.execute("select 1 from sqlite_master where type='table' and name=?",(table,)).fetchone() else 0
    assert all(count(table)==0 for table in ("hypotheses","experiments","candidates","data_partitions","research_runs"))
    paths=client.get("/openapi.json").json()["paths"];assert not any(any(token in path for token in ("validation","final","order","shell")) for path in paths)

def test_certified_and_duplicate_diagnostics_are_structured(client):
    certified=upload(client).json();assert certified["certification"]=="CERTIFIED" and certified["validation_summary"]["invalid_ohlc_count"]==0 and certified["quarantine_reasons"]==[]
    duplicate=upload(client).json();assert duplicate["certification"]=="DUPLICATE" and duplicate["dataset_id"]==certified["dataset_id"] and duplicate["original_sha256"]==certified["original_sha256"]

def test_quarantine_diagnostics_include_bounded_safe_ohlc_examples(client):
    rows=[{"timestamp":f"2024-01-01 00:{i:02d}","open":1.0,"high":.9,"low":.8,"close":1.1} for i in range(10)];result=upload(client,"invalid.csv",csv_bytes(rows)).json()
    assert result["certification"]=="QUARANTINED" and result["validation_summary"]["invalid_ohlc_count"]==10 and len(result["quarantine_reasons"])==1
    finding=result["quarantine_reasons"][0];assert finding["error_code"]=="INVALID_OHLC" and finding["count"]==10 and 1<=len(finding["examples"])<=5 and finding["first_affected_timestamp"]
    assert all(set(x["values"])=={"open","high","low","close"} for x in finding["examples"])

def test_rejected_schema_and_timestamp_diagnostics_are_safe(client):
    schema=upload(client,body=b"when,value\n2024-01-01,1\n").json();timestamp=upload(client,body=csv_bytes([{"timestamp":"not-a-date","open":1,"high":2,"low":.5,"close":1.5}])).json()
    assert schema["certification"]=="REJECTED" and schema["dataset_id"] is None and schema["validation_summary"]["schema_error_count"]==1
    assert timestamp["certification"]=="REJECTED" and timestamp["validation_summary"]["invalid_timestamp_count"]==1
    text=json.dumps([schema,timestamp]).lower();assert not any(token in text for token in ("traceback","tests\\runtime","sqlite","parquet","password","final_test","validation_context"))

def test_alignment_and_gap_diagnostics_are_visible(client):
    base=pd.Timestamp("2024-01-01",tz="UTC");values=[{"timestamp":int((base+pd.Timedelta(minutes=m)).timestamp()*1000),"open":1,"high":2,"low":.5,"close":1.5} for m in (7,30,60)]
    result=upload(client,"m15.csv",pd.DataFrame(values).to_csv(index=False).encode(),data_type="M15",timestamp_format="UNIX_MILLISECONDS").json();summary=result["validation_summary"]
    assert result["certification"]=="QUARANTINED" and summary["misaligned_timestamp_count"]==1 and summary["suspicious_gap_count"]==2 and summary["unexpected_gap_count"]==2
    assert {x["error_code"] for x in result["quarantine_reasons"]}>={"MISALIGNED_M15_TIMESTAMP","SUSPICIOUS_GAP"}

def test_previous_import_diagnostics_are_selectable_and_do_not_expose_paths(client):
    created=upload(client,"invalid.csv",csv_bytes([{"timestamp":"2024-01-01","open":1,"high":.9,"low":.8,"close":1.1}])).json();response=client.get(f'/api/data/imports/{created["dataset_id"]}/diagnostics')
    assert response.status_code==200 and response.json()["validation_summary"]["invalid_ohlc_count"]==1 and response.json()["certification"]=="QUARANTINED"
    assert "path" not in response.text.lower() and client.get("/api/data/imports/../../catalog.sqlite3/diagnostics").status_code in (404,422)

def test_diagnostics_add_no_bypass_or_research_state(client,root):
    created=upload(client,"invalid.csv",csv_bytes([{"timestamp":"2024-01-01","open":1,"high":.9,"low":.8,"close":1.1}])).json();client.get(f'/api/data/imports/{created["dataset_id"]}/diagnostics');con=sqlite3.connect(root/"catalog.sqlite3")
    def count(table):return con.execute("select count(*) from "+table).fetchone()[0] if con.execute("select 1 from sqlite_master where type='table' and name=?",(table,)).fetchone() else 0
    assert all(count(table)==0 for table in ("hypotheses","experiments","candidates","data_partitions","research_runs")) and count("datasets")==0
    paths=client.get("/openapi.json").json()["paths"];assert not any(any(token in path.lower() for token in ("force","certify","delete","validation","final","order","shell")) for path in paths)

def reinterpretation_bytes(invalid=False):
    values=[{"timestamp":1704067200000,"open":1.0,"high":.9 if invalid else 1.2,"low":.8,"close":1.1},{"timestamp":1704068100000,"open":1.1,"high":1.3,"low":1.0,"close":1.2}];return pd.DataFrame(values).to_csv(index=False).encode()

def test_same_bytes_same_metadata_remain_duplicate_even_with_reason(client):
    body=reinterpretation_bytes();first=upload(client,"source.csv",body,data_type="M15",timestamp_format="UNIX_MILLISECONDS").json();duplicate=upload(client,"source.csv",body,data_type="M15",timestamp_format="UNIX_MILLISECONDS",interpretation_reason="A reason cannot create a new identity for identical metadata.").json()
    assert first["certification"]=="CERTIFIED" and duplicate["certification"]=="DUPLICATE" and duplicate["dataset_id"]==first["dataset_id"] and duplicate["dataset_fingerprint"]==first["dataset_fingerprint"]

def test_multiple_interpretations_require_reason_then_preserve_reason_and_old_record(client,root):
    body=reinterpretation_bytes();first=upload(client,"source.csv",body,data_type="M15",timestamp_format="UNIX_MILLISECONDS").json();con=sqlite3.connect(root/"catalog.sqlite3");before=con.execute("select manifest_json from external_imports where dataset_id=?",(first["dataset_id"],)).fetchone()[0]
    missing=upload(client,"source.csv",body,data_type="M1",timestamp_format="UNIX_MILLISECONDS").json();empty=upload(client,"source.csv",body,data_type="M1",timestamp_format="UNIX_MILLISECONDS",interpretation_reason="   ").json()
    assert missing["certification"]==empty["certification"]=="REJECTED" and missing["rejection_reasons"][0]["error_code"]=="MULTIPLE_INTERPRETATIONS_REASON_REQUIRED"
    reason="Corrected synthetic source interpretation after documented metadata review."
    corrected=upload(client,"source.csv",body,data_type="M1",timestamp_format="UNIX_MILLISECONDS",interpretation_reason=f"  {reason}  ").json();assert corrected["certification"]=="CERTIFIED" and corrected["dataset_id"]!=first["dataset_id"] and corrected["dataset_fingerprint"]!=first["dataset_fingerprint"]
    manifest=json.loads(con.execute("select manifest_json from external_imports where dataset_id=?",(corrected["dataset_id"],)).fetchone()[0]);assert manifest["multiple_interpretations"] is True and manifest["interpretation_reason"]==reason
    assert con.execute("select manifest_json from external_imports where dataset_id=?",(first["dataset_id"],)).fetchone()[0]==before

def test_interpretation_reason_never_bypasses_quarantine(client):
    body=reinterpretation_bytes(invalid=True);first=upload(client,"source.csv",body,data_type="M15",timestamp_format="UNIX_MILLISECONDS").json();assert first["certification"]=="QUARANTINED"
    reinterpreted=upload(client,"source.csv",body,data_type="M1",timestamp_format="UNIX_MILLISECONDS",interpretation_reason="Different timeframe interpretation retained for a quarantine safety test.").json();assert reinterpreted["certification"]=="QUARANTINED" and reinterpreted["validation_summary"]["invalid_ohlc_count"]==1

@pytest.mark.parametrize("reason",["x"*501,"<script>alert(1)</script>","../../secret","C:\\secret\\file"])
def test_interpretation_reason_length_and_unsafe_path_markup_are_rejected(client,reason):
    response=upload(client,"source.csv",reinterpretation_bytes(),data_type="M15",timestamp_format="UNIX_MILLISECONDS",interpretation_reason=reason);assert response.status_code==422 and "Traceback" not in response.text and "secret" not in response.text
