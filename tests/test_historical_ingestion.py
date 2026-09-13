from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import json
import sqlite3
import shutil
import pandas as pd
import pytest
from sandbox.catalog import Catalog
from sandbox.cli import main, parser
from sandbox.config import Settings
from sandbox.external_import import ExternalImportGateway, ImportSpec, ImportError, Resolution, PriceType
from sandbox.market_state.universal.dataset import validate_bars, export_discovery_features
from sandbox.partition.service import PartitionService
from sandbox.research.registry import ResearchRegistry
from sandbox.provenance import Provenance


@pytest.fixture
def root():
    p=Path("tests/runtime")/("step15-"+uuid4().hex)
    p.mkdir(parents=True)
    shutil.copytree("config",p/"config")
    return p.resolve()


@pytest.fixture
def real_bars():
    # Accepted local broker fixture, not the quarantined external CSV.
    path=Path("data/raw/ic_markets/ICMarketsEU-Demo/EURUSD/M1/2026/08.parquet")
    if not path.exists():
        pytest.skip("accepted 61-bar real broker fixture unavailable")
    return pd.read_parquet(path)


def m15(frame):
    request=SimpleNamespace(metadata=SimpleNamespace(required_timeframes=("M15",)))
    return PartitionService._strategy_timeframe_frames(frame, request, "M1")["M15"]


def test_cli_m15_unix_mapping_runs_normal_validation(root, real_bars, monkeypatch, capsys):
    frame=m15(real_bars)
    frame["timestamp"]=[int(t.timestamp()*1000) for t in frame.timestamp_utc]
    source=root/"real-m15.csv"
    frame.to_csv(source,index=False)
    monkeypatch.setattr(Settings,"load",classmethod(lambda cls:Settings(None,root/"data",root/"catalog.sqlite3",7,3,"ERROR")))
    result=main(["data","import-file","--file",str(source),"--provider","IC_MARKETS",
                 "--symbol","EURUSD","--resolution","M15","--price-type","BID","--timezone","UTC",
                 "--timestamp-format","UNIX_MILLISECONDS"])
    assert result == 0
    response=json.loads(capsys.readouterr().out)
    assert response["status"] == "STAGED"
    gateway=ExternalImportGateway(Catalog(root/"catalog.sqlite3"),root/"data")
    record=gateway.inspect_dataset(response["dataset_id"])
    assert record["manifest"]["row_count"] == 4
    assert record["manifest"]["column_mapping"] == {k:k for k in ("timestamp","open","high","low","close")}
    assert record["manifest"]["timestamp_format"] == "UNIX_MILLISECONDS"
    assert gateway.certify(response["dataset_id"])["status"] == "CERTIFIED"


def test_cli_default_timestamp_format_is_backward_compatible():
    args=parser().parse_args(["data","import-file","--file","x.csv","--provider","X","--symbol","EURUSD",
        "--resolution","M1","--price-type","BID","--timezone","UTC"])
    assert args.timestamp_format == "ISO8601"


@pytest.mark.parametrize("problem", ["duplicate","unordered","naive","offgrid","nonfinite","negative","geometry"])
def test_preflight_rejects_corruption_without_relaxing_rules(real_bars, problem):
    f=real_bars.copy()
    if problem=="duplicate":f.loc[1,"timestamp_utc"]=f.timestamp_utc.iloc[0]
    if problem=="unordered":f=f.iloc[::-1]
    if problem=="naive":f.timestamp_utc=f.timestamp_utc.dt.tz_localize(None)
    if problem=="offgrid":f.timestamp_utc += pd.Timedelta("1s")
    if problem=="nonfinite":f.loc[0,"high"]=float("inf")
    if problem=="negative":f.loc[0,"low"]=-1
    if problem=="geometry":f.loc[0,"high"]=f.low.iloc[0]-.01
    with pytest.raises(ValueError):validate_bars(f,"M1")


def test_real_accepted_source_and_complete_derivation(real_bars):
    q=validate_bars(real_bars,"M1")
    assert q["row_count"] == 61 and q["missing_grid_slots"] == 0
    derived=m15(real_bars)
    assert validate_bars(derived,"M15")["row_count"] == 4
    assert derived.open.iloc[0] == real_bars.open.iloc[0]
    assert derived.close.iloc[0] == real_bars.close.iloc[14]


def test_quarantine_cannot_be_certified_or_partitioned(root,real_bars):
    f=real_bars.copy()
    f.loc[0,"high"]=f.low.iloc[0]-.01
    source=root/"invalid-real-bar.csv";f.to_csv(source,index=False)
    g=ExternalImportGateway(Catalog(root/"catalog.sqlite3"),root/"data")
    result=g.import_files(ImportSpec((source,),"IC_MARKETS","EURUSD","EURUSD",Resolution.M1,
        PriceType.BID,"UTC",dict(timestamp="timestamp_utc",open="open",high="high",low="low",close="close")))
    assert result["status"] == "QUARANTINED"
    with pytest.raises(ImportError,match="QUARANTINED"):g.certify(result["dataset_id"])
    service=PartitionService(ResearchRegistry(root/"catalog.sqlite3",root))
    with pytest.raises(Exception,match="not registered"):
        service.create_partition(result["dataset_id"],"INGESTION","DISCOVERY","EURUSD","M1",
                                 "2026-08-27T08:00:00Z","2026-08-27T09:01:00Z")


def test_real_discovery_export_refuses_unwarmed_data_and_protected_role(root,real_bars):
    source=root/"source.parquet";real_bars.to_parquet(source,index=False)
    registry=ResearchRegistry(root/"catalog.sqlite3",root)
    provenance=Provenance.create(source,"IC Markets (EU) Ltd","ICMarketsEU-Demo","EURUSD","M1",
                                real_bars.timestamp_utc.iloc[0],real_bars.timestamp_utc.iloc[-1],len(real_bars))
    registry.catalog.add_dataset(provenance)
    service=PartitionService(registry)
    for role in ("DISCOVERY","VALIDATION"):
        part=service.create_partition(provenance.dataset_id,"INGESTION_"+role,role,"EURUSD","M1",
                                      "2026-08-27T08:00:00Z","2026-08-27T09:01:00Z")
        with pytest.raises(Exception,match="INSUFFICIENT_READY_FEATURE_ROWS" if role=="DISCOVERY" else "ACCESS_DENIED"):
            export_discovery_features(service,part.partition_id)
    assert not (root/"data/derived/features").exists()
    assert service.verify_access_ledger()
