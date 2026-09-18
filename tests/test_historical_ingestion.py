from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4
import json
import sqlite3
import shutil
import pandas as pd
import pytest
import sandbox.cli as cli_module
from sandbox.catalog import Catalog
from sandbox.cli import main, parser
from sandbox.config import Settings
from sandbox.external_import import ExternalImportGateway, ImportSpec, ImportError, Resolution, PriceType
from sandbox.market_state.universal.dataset import validate_bars, export_discovery_features
from sandbox.partition.service import PartitionService
from sandbox.research.registry import ResearchRegistry
from sandbox.provenance import Provenance
from sandbox.models import SymbolRecord


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
    frame = pd.read_parquet(path)
    # The canonical month grows as history is ingested; this fixture is the
    # original one-hour sample, not whichever month happens to be installed.
    times = pd.to_datetime(frame.timestamp_utc, utc=True)
    sample = frame.loc[times.between("2026-08-27T08:00:00Z", "2026-08-27T09:00:00Z")]
    return sample.reset_index(drop=True)


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


def test_cli_history_download_retry_merges_and_reuses_catalog_dataset(root,monkeypatch,capsys):
    partial=pd.DataFrame({"timestamp_utc":pd.to_datetime(["2024-01-01T00:00:00Z"],utc=True),
        "open":[1.1],"high":[1.2],"low":[1.0],"close":[1.15],"tick_volume":[1],"spread":[1],"real_volume":[0]})
    expanded=pd.concat([partial,pd.DataFrame({"timestamp_utc":pd.to_datetime(["2024-01-01T00:01:00Z"],utc=True),
        "open":[1.1],"high":[1.2],"low":[1.0],"close":[1.15],"tick_volume":[1],"spread":[1],"real_volume":[0]})],ignore_index=True)
    frames=iter((partial,expanded))
    class Adapter:
        def status(self):return SimpleNamespace(company="IC Markets",server="TEST-SERVER")
        def resolve_symbol(self,pair):return SymbolRecord(broker_symbol="EURUSD",canonical_pair="EURUSD",digits=5,point=.00001,visible=True)
        def shutdown(self):pass
    monkeypatch.setattr(Settings,"load",classmethod(lambda cls:Settings(None,root/"data",root/"catalog.sqlite3",7,3,"ERROR")))
    monkeypatch.setattr(cli_module,"_connect",lambda settings:Adapter())
    monkeypatch.setattr(cli_module,"download",lambda *args,**kwargs:next(frames))
    command=["history","download","EURUSD","--start","2024-01-01T00:00:00Z","--end","2024-01-01T00:02:00Z"]
    assert main(command)==0;first=json.loads(capsys.readouterr().out)
    assert main(command)==0;second=json.loads(capsys.readouterr().out)
    assert first[0]["dataset_id"]==second[0]["dataset_id"]
    assert second[0]["row_count"]==2
    assert len(Catalog(root/"catalog.sqlite3").datasets())==1


@pytest.mark.parametrize("problem", ["duplicate","unordered","naive","offgrid","nonfinite","negative","geometry"])
def test_preflight_rejects_corruption_without_relaxing_rules(real_bars, problem):
    f=real_bars.copy()
    if problem=="duplicate":f.loc[1,"timestamp_utc"]=f.timestamp_utc.iloc[0]
    if problem=="unordered":f=f.iloc[::-1]
    if problem=="naive":f.timestamp_utc=f.timestamp_utc.dt.tz_localize(None)
    if problem=="offgrid":f.timestamp_utc += timedelta(seconds=1)
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


def test_raw_partition_adds_continuity_boundary_at_missing_m1_bar(root):
    source = root / "gapped.parquet"
    bars = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z",
            "2024-01-01T00:03:00Z", "2024-01-01T00:04:00Z",
        ], utc=True),
        "open": [1.1] * 4, "high": [1.2] * 4, "low": [1.0] * 4,
        "close": [1.15] * 4, "tick_volume": [1] * 4,
        "spread": [1] * 4, "real_volume": [0] * 4,
    })
    bars.to_parquet(source, index=False)
    registry = ResearchRegistry(root / "catalog.sqlite3", root)
    provenance = Provenance.create(source, "IC Markets (EU) Ltd", "ICMarketsEU-Demo",
                                   "EURUSD", "M1", bars.timestamp_utc.iloc[0],
                                   bars.timestamp_utc.iloc[-1], len(bars))
    registry.catalog.add_dataset(provenance)
    service = PartitionService(registry)
    partition = service.create_partition(provenance.dataset_id, "INGESTION", "DISCOVERY",
                                         "EURUSD", "M1", "2024-01-01T00:00:00Z",
                                         "2024-01-01T00:05:00Z")
    with registry.catalog.connection() as con:
        path = Path(con.execute("SELECT path FROM data_partitions WHERE partition_id=?",
                                (partition.partition_id,)).fetchone()[0])
    partition_bars = pd.read_parquet(path)
    assert partition_bars.continuity_segment_id.tolist() == [0, 0, 1, 1]
    pd.testing.assert_frame_equal(partition_bars[list(bars.columns)], bars)


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
