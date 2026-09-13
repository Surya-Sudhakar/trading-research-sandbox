from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sandbox.api.app import create_app
from sandbox.api.service import SandboxReadService
from sandbox.catalog import Catalog
from sandbox.config import Settings
from sandbox.external_import import ExternalImportGateway, ImportError, ImportSpec, PriceType, Resolution, TimestampFormat
from sandbox.market_state import FeatureConfiguration, MarketStateEngine
from sandbox.partition.service import PartitionService
from sandbox.provenance import sha256_file
from sandbox.strategies.s001 import S001Strategy


@pytest.fixture
def root():
    path=(Path("tests/runtime")/("m15-import-"+uuid4().hex)).resolve();path.mkdir(parents=True);return path


def millis(value: str) -> int:return int(pd.Timestamp(value,tz="UTC").timestamp()*1000)
def rows(times=("2024-01-01 00:00","2024-01-01 00:15","2024-01-01 00:30","2024-01-01 00:45")):
    return [{"timestamp":millis(t),"open":1.10+i*.01,"high":1.12+i*.01,"low":1.09+i*.01,"close":1.11+i*.01} for i,t in enumerate(times)]
def write(path: Path, values=None):pd.DataFrame(values or rows()).to_csv(path,index=False);return path
def spec(path: Path, resolution=Resolution.M15, timestamp_format=TimestampFormat.UNIX_MILLISECONDS):
    return ImportSpec((path,),"DUKASCOPY","EURUSD","EURUSD",resolution,PriceType.BID,"UTC",{"timestamp":"timestamp","open":"open","high":"high","low":"low","close":"close"},None,False,None,timestamp_format)
def gateway(root):return ExternalImportGateway(Catalog(root/"catalog.sqlite3"),root/"data")
def client(root):return TestClient(create_app(SandboxReadService(Settings(None,root/"data",root/"catalog.sqlite3",7,3,"INFO"),root,probe_broker=False)))
def upload(client,values=None,**overrides):
    metadata={"provider":"DUKASCOPY","symbol":"EURUSD","data_type":"M15","price_type":"BID","source_timezone":"UTC","purpose":"UNASSIGNED_RESEARCH_DATA","timestamp_format":"UNIX_MILLISECONDS"};metadata.update(overrides)
    return client.post("/api/data/import",data=metadata,files={"file":("m15.csv",pd.DataFrame(values or rows()).to_csv(index=False).encode(),"text/csv")})


def test_valid_m15_unix_ms_utc_provenance_sha_and_catalog(root):
    source=write(root/"m15.csv");original=sha256_file(source);g=gateway(root);result=g.import_files(spec(source));assert result["status"]=="STAGED"
    record=g.inspect_dataset(result["dataset_id"]);manifest=record["manifest"]
    assert manifest["resolution"]=="M15" and manifest["timestamp_format"]=="UNIX_MILLISECONDS" and manifest["source_timezone"]=="UTC"
    assert manifest["price_type"]=="BID" and manifest["provider_authentication"]=="DECLARED_PROVENANCE" and manifest["component_sha256s"]==[original]
    assert manifest["first_timestamp"]=="2024-01-01T00:00:00+00:00";g.certify(result["dataset_id"])
    assert g.catalog.datasets()[0]["timeframe"]=="M15"


@pytest.mark.parametrize("minute",[0,15,30,45])
def test_each_m15_boundary_is_valid(root,minute):
    result=gateway(root).import_files(spec(write(root/f"m15-{minute}.csv",rows((f"2024-01-01 00:{minute:02d}",)))));assert result["status"]=="STAGED"


def test_misaligned_m15_is_quarantined_without_rounding(root):
    g=gateway(root);source=write(root/"bad.csv",rows(("2024-01-01 00:07",)));result=g.import_files(spec(source));assert result["status"]=="QUARANTINED"
    record=g.inspect_dataset(result["dataset_id"]);assert record["manifest"]["first_timestamp"]=="2024-01-01T00:07:00+00:00" and result["audit"]["rows_rejected"]==1
    assert any(x["code"]=="misaligned_m15_timestamp" for x in result["audit"]["issues"])


@pytest.mark.parametrize("value",["not-ms",float("nan"),float("inf"),1.5,10**30])
def test_invalid_unix_milliseconds_are_rejected(root,value):
    bad=rows(("2024-01-01 00:00",));bad[0]["timestamp"]=value
    with pytest.raises(ImportError,match="no valid rows"):gateway(root).import_files(spec(write(root/"bad.csv",bad)))


def test_duplicate_geometry_and_nonfinite_m15_quarantine(root):
    values=rows(("2024-01-01 00:00","2024-01-01 00:00","2024-01-01 00:15"));values[0]["high"]=values[0]["low"]-.01;values[2]["close"]=float("inf")
    result=gateway(root).import_files(spec(write(root/"bad.csv",values)));assert result["status"]=="QUARANTINED" and result["audit"]["duplicate_events"]>=1 and result["audit"]["ohlc_violations"]>=1 and result["audit"]["rows_rejected"]>=2


def test_m15_gap_semantics_weekend_and_no_filling(root):
    g=gateway(root);normal=g.import_files(spec(write(root/"normal.csv",rows())));assert normal["audit"]["suspicious_gaps"]==0
    missing=g.import_files(spec(write(root/"missing.csv",rows(("2024-01-01 00:00","2024-01-01 00:30")))));assert missing["audit"]["suspicious_gaps"]==1 and g.inspect_dataset(missing["dataset_id"])["manifest"]["row_count"]==2
    weekend=g.import_files(spec(write(root/"weekend.csv",rows(("2024-01-05 21:45","2024-01-07 22:00")))));assert weekend["audit"]["expected_gaps"]==1


def test_m15_cannot_be_silently_relabelled_m1(root):
    source=write(root/"same.csv");g=gateway(root);m15=g.import_files(spec(source));m1_spec=spec(source,Resolution.M1);m1_spec=ImportSpec(**{**m1_spec.__dict__,"interpretation_reason":"Explicit comparison proves timeframe is fingerprint material"});m1=g.import_files(m1_spec);assert m15["dataset_id"]!=m1["dataset_id"]
    assert g.inspect_dataset(m15["dataset_id"])["manifest"]["resolution"]=="M15" and g.inspect_dataset(m1["dataset_id"])["manifest"]["resolution"]=="M1"


def test_api_accepts_m15_ohlc_and_blocks_protected_purposes(root):
    c=client(root);response=upload(c);assert response.status_code==200 and response.json()["data_type"]=="M15" and response.json()["certification"]=="CERTIFIED"
    assert c.get("/api/data/status").json()["datasets"][0]["timeframe"]=="M15"
    assert upload(c,purpose="VALIDATION").status_code==422 and upload(c,purpose="FINAL_TEST").status_code==422
    con=sqlite3.connect(root/"catalog.sqlite3");assert con.execute("select count(*) from sqlite_master where type='table' and name='data_partitions'").fetchone()[0]==0


def test_m15_builds_only_complete_causal_h1_for_s001_and_market_state():
    frame=pd.DataFrame(rows());frame["timestamp_utc"]=pd.to_datetime(frame.pop("timestamp"),unit="ms",utc=True);frame["tick_volume"]=0;frame["spread"]=0;frame["real_volume"]=0
    plugin=S001Strategy();frames=PartitionService._strategy_timeframe_frames(frame,plugin,"M15");assert frames["M15"].equals(frame) and len(frames["H1"])==1
    incomplete=PartitionService._strategy_timeframe_frames(frame.iloc[:3],plugin,"M15");assert incomplete["H1"].empty
    engine=MarketStateEngine(FeatureConfiguration(requested_timeframes=("M15","H1"),horizons=(1,),atr_period=1,volatility_short_window=1,volatility_long_window=2,volatility_percentile_history=2))
    snapshot=engine.snapshot(frames,symbol="EURUSD",decision_timestamp=pd.Timestamp("2024-01-01 01:00",tz="UTC").to_pydatetime());assert snapshot.information_cutoff_timestamp.hour==1


def test_frontend_exposes_controlled_m15_and_timestamp_formats():
    source=Path("frontend/src/App.tsx").read_text(encoding="utf-8");assert "<option>M15</option>" in source and "UNIX_MILLISECONDS" in source and "ISO8601" in source
    assert all(text in source for text in ("<<< Import Diagnostics >>>","<<< Findings >>>","<<< Representative Examples >>>","INSPECT SAFE DIAGNOSTICS","NOT ELIGIBLE FOR RESEARCH"))
    assert all(text in source for text in ("INTERPRETATION REASON","MULTIPLE_INTERPRETATIONS_REASON_REQUIRED","not a validation bypass","maxLength={500}"))
