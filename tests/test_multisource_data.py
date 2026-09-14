from pathlib import Path
from datetime import timedelta
from uuid import uuid4
import shutil
import pandas as pd
import pytest
from sandbox.market_data.models import ProviderMetadata,CanonicalMarketBar
from sandbox.market_data.sanitation import sanitize,frame_hash
from sandbox.market_data.gaps import SessionProfile,ContinuityPolicy,inventory
from sandbox.market_data.storage import sanitize_external,lineage_record,validation_view
from sandbox.market_data.archive import MT5Archive,merge_archive
from sandbox.market_data.comparison import compare_providers
from sandbox.market_data.providers import ExternalCSVAdapter
from sandbox.market_state.universal import UniversalFeatureEngine
from sandbox.market_state.universal.continuity import QualityContext
from sandbox.market_state.universal.dataset import validate_bars,export_discovery_features
from sandbox.external_import import ExternalImportGateway,ImportSpec,Resolution,PriceType
from sandbox.research.registry import ResearchRegistry
from sandbox.partition.service import PartitionService

META=ProviderMetadata("TEST_PROVIDER","EURUSD","EURUSD","M15","BID","RAW_TEST")


def bars(n=160,start="2024-01-01",freq="15min"):
    base=[1.1+i*.00001 for i in range(n)]
    return pd.DataFrame(dict(timestamp_utc=pd.date_range(start,periods=n,freq=freq,tz="UTC"),
                             open=base,high=[x+.00004 for x in base],low=[x-.00002 for x in base],
                             close=[x+.00001 for x in base]))


@pytest.fixture
def root():
    p=Path("tests/runtime")/("multisource-"+uuid4().hex);p.mkdir(parents=True)
    shutil.copytree("config",p/"config")
    return p.resolve()


def test_invalid_exclusion_no_correction_and_raw_unchanged():
    raw=bars();raw.loc[10,"high"]=raw.open.iloc[10]-.00001
    original=raw.copy(deep=True)
    clean,audit,gaps,m=sanitize(raw,META)
    pd.testing.assert_frame_equal(raw,original)
    assert len(clean)==159 and m["exclusions_by_reason"]=={"INVALID_OHLC":1}
    assert audit.loc[10,"classification"]=="INVALID_OHLC"
    assert str(raw.open.iloc[10]) in audit.loc[10,"original_observation"]
    expected=raw.drop(index=10).reset_index(drop=True)
    pd.testing.assert_frame_equal(clean[list(expected)],expected)
    assert gaps.iloc[0].invalid_bar_exclusion and gaps.iloc[0].reset


@pytest.mark.parametrize("column,value,reason",[
    ("high",float("inf"),"NONFINITE_PRICE"),("low",0,"NONPOSITIVE_PRICE"),
    ("timestamp_utc",pd.Timestamp("2024-01-01 00:01Z"),"OFF_GRID"),
    ("timestamp_utc",pd.NaT,"INVALID_TIMESTAMP"),("spread",-1,"OTHER_INVALID")])
def test_invalid_classes_are_audited(column,value,reason):
    raw=bars()
    raw.loc[0,column]=value
    clean,audit,gaps,m=sanitize(raw,META)
    assert audit.loc[0,"classification"]==reason and m["excluded_row_count"]==1
    if reason=="INVALID_TIMESTAMP":
        assert clean.continuity_segment_id.nunique()==len(clean)


def test_identical_duplicates_keep_first_and_conflicts_exclude_all():
    raw=bars(5)
    duplicate=pd.concat([raw,raw.iloc[[1]]],ignore_index=True)
    clean,audit,_,m=sanitize(duplicate,META)
    assert len(clean)==5 and audit.classification.iloc[-1]=="DUPLICATE_IDENTICAL"
    duplicate.loc[5,"close"]+=.000001
    clean,audit,_,m=sanitize(duplicate,META)
    assert len(clean)==4 and m["exclusions_by_reason"]=={"DUPLICATE_CONFLICT":2}
    assert raw.timestamp_utc.iloc[1] not in set(clean.timestamp_utc)


def test_sanitation_reproducibility():
    raw=bars();raw.loc[10,"low"]=2
    a=sanitize(raw,META);b=sanitize(raw,META)
    for i in range(3):pd.testing.assert_frame_equal(a[i],b[i])
    assert a[3]==b[3]


def test_profile_requires_evidence():
    with pytest.raises(ValueError):SessionProfile(week_close=(4,17,0),week_open=(6,17,0))
    with pytest.raises(ValueError):SessionProfile(week_close=(4,17,0))


def test_explicit_weekend_dst_and_unknown_session():
    known=SessionProfile("TEST_NY","America/New_York",(4,17,0),(6,17,0),"test fixture schedule")
    left=pd.Timestamp("2024-03-08 21:45Z");right=pd.Timestamp("2024-03-10 21:00Z")
    assert known.classify(left,right,15)=="EXPECTED_WEEKEND"
    assert SessionProfile().classify(left,right,15)=="WEEKEND_ASSOCIATED_UNKNOWN"
    assert known.classify(left,pd.Timestamp("2024-03-11 00:00Z"),15)=="WEEKEND_ASSOCIATED_UNKNOWN"


def test_continuity_resets_unexplained_and_can_continue_expected_weekend():
    raw=bars();raw.loc[80:,"timestamp_utc"]+=timedelta(hours=4)
    clean,_,gaps,_=sanitize(raw,META)
    assert clean.continuity_segment_id.nunique()==2 and gaps.iloc[0].reset
    assert clean.continuity_segment_id.iloc[79]==0 and clean.continuity_segment_id.iloc[80]==1
    known=SessionProfile("TEST_NY","America/New_York",(4,17,0),(6,17,0),"test fixture schedule")
    weekend=bars(2);weekend.timestamp_utc=pd.to_datetime(["2024-03-08 21:45Z","2024-03-10 21:00Z"])
    assert sanitize(weekend,META,known)[0].continuity_segment_id.nunique()==1
    assert sanitize(weekend,META,known,ContinuityPolicy(continue_expected_weekend=False))[0].continuity_segment_id.nunique()==2


def test_feature_optional_quality_no_reset_regression_and_unexplained_reset():
    engine=UniversalFeatureEngine();raw=bars(280)
    clean,_,_,_=sanitize(raw,META)
    assert engine.compute(clean,symbol="EURUSD")==engine.compute(clean,symbol="EURUSD",quality_context=QualityContext())
    raw.loc[140:,"timestamp_utc"]+=timedelta(hours=4)
    clean,_,_,_=sanitize(raw,META)
    rows=engine.compute(clean,symbol="EURUSD",quality_context=QualityContext())
    assert rows[139].feature_ready and not rows[140].feature_ready
    assert rows[140].atr_14 is None and rows[140].prior_high_12 is None
    assert rows[140].completed_h1_direction is None and rows[140].completed_h3_direction is None
    assert rows[253].feature_ready
    changed=clean.copy();changed.loc[200:,["open","high","low","close"]]=[2,3,1,1.5]
    assert rows[:200]==engine.compute(changed,symbol="EURUSD",quality_context=QualityContext())[:200]


def test_canonical_optional_unknown_market_fields():
    clean,_,_,_=sanitize(bars(1),META)
    row=clean.iloc[0].to_dict()
    for name in ("spread","tick_volume","real_volume"):row[name]=None
    bar=CanonicalMarketBar(**row)
    assert bar.spread is None and bar.provider=="TEST_PROVIDER"
    assert validate_bars(clean,"M15")["row_count"]==1


def import_raw(root):
    raw=bars(280);raw.loc[140,"high"]=1
    source=root/"raw.csv";raw.rename(columns={"timestamp_utc":"timestamp"}).to_csv(source,index=False)
    registry=ResearchRegistry(root/"catalog.sqlite3",root)
    gateway=ExternalImportGateway(registry.catalog,root/"data")
    result=gateway.import_files(ImportSpec((source,),"TEST_PROVIDER","EURUSD","EURUSD",Resolution.M15,
        PriceType.BID,"UTC",{k:k for k in ("timestamp","open","high","low","close")}))
    return registry,gateway,result["dataset_id"],source


def test_raw_immutability_manifest_parent_lineage_and_idempotent_registration(root):
    registry,gateway,rawid,path=import_raw(root);original=path.read_bytes()
    first,data=sanitize_external(registry,rawid);second,_=sanitize_external(registry,rawid)
    assert first==second and path.read_bytes()==original
    assert gateway.inspect_dataset(rawid)["status"]=="QUARANTINED"
    assert first["parent_raw_dataset_id"]==rawid and first["kind"]=="SANITIZED"
    assert len(registry.catalog.datasets())==1
    assert lineage_record(registry,first["sanitized_dataset_id"])==first
    registry.verify_dataset(first["sanitized_dataset_id"])


def test_sanitized_partition_controls_and_feature_lineage(root):
    registry,gateway,rawid,path=import_raw(root);m,_=sanitize_external(registry,rawid)
    service=PartitionService(registry)
    with pytest.raises(Exception,match="not registered"):
        service.create_partition(rawid,"RAW","DISCOVERY","EURUSD","M15","2024-01-01T00:00Z","2024-01-04T00:00Z")
    for role in ("DISCOVERY","VALIDATION","FINAL_TEST"):
        p=service.create_partition(m["sanitized_dataset_id"],role,role,"EURUSD","M15","2024-01-01T00:00Z","2024-01-04T00:00Z")
        if role=="DISCOVERY":
            result=export_discovery_features(service,p.partition_id)
            output=pd.read_parquet(root/result["artifact_path"])
            assert result["quality_mode"]=="SEGMENT_LOCAL_V1" and output.feature_ready.any()
            assert output.continuity_segment_id.nunique()==2
            assert output.source_dataset_id.eq(m["sanitized_dataset_id"]).all()
            assert result==export_discovery_features(service,p.partition_id)
        else:
            with pytest.raises(Exception,match="ACCESS_DENIED"):export_discovery_features(service,p.partition_id)


def test_archive_merge_conflicts_idempotence_restart_and_completion(root):
    raw=bars(100,freq="min");archive=MT5Archive(root/"archive")
    first=archive.ingest(raw.iloc[:60],as_of=pd.Timestamp("2024-01-01 02:00Z"))
    paths=list(archive.root.glob("????/??/*.parquet"));old={p:p.read_bytes() for p in paths}
    assert first["new_rows"]==60
    second=MT5Archive(root/"archive").ingest(raw.iloc[50:],as_of=pd.Timestamp("2024-01-01 01:39Z"))
    assert second["total_rows"]==99
    assert all(p.read_bytes()==value for p,value in old.items())
    assert archive.ingest(raw.iloc[:99],as_of=pd.Timestamp("2024-01-01 02:00Z"))["new_rows"]==0
    broken=raw.iloc[:1].copy();broken["close"]+=.000001
    with pytest.raises(ValueError,match="CONFLICTING"):archive.ingest(broken,as_of=pd.Timestamp("2024-01-01 02:00Z"))
    assert len(archive.read())==99


def test_archive_poll_overlap_is_incremental(root):
    raw=bars(100,freq="min");archive=MT5Archive(root/"archive")
    archive.ingest(raw,as_of=pd.Timestamp("2024-01-01 02:00Z"))
    class Feed:
        def rates(self,symbol,start,end):
            self.start=start
            return raw[raw.timestamp_utc>=start]
    feed=Feed()
    archive.poll(feed,now=pd.Timestamp("2024-01-01 02:00Z"),overlap_minutes=10)
    assert feed.start==raw.timestamp_utc.max()-timedelta(minutes=10)


def test_cross_provider_alignment_variance_and_no_overlap():
    a=bars(5);b=bars(5).iloc[1:].copy()
    b[["open","high","low","close"]]+=.0001
    rows,m=compare_providers(a,b)
    assert m["matched_bars"]==4 and m["missing_on_b"]==1
    assert m["median_absolute_close_difference_pips"]==pytest.approx(1)
    assert m["direction_agreement_rate"]==1
    b.timestamp_utc+=timedelta(days=7)
    assert compare_providers(a,b)[1]["matched_bars"]==0

def test_sanitation_restart_recovers_complete_unregistered_artifact(root):
    registry,gateway,rawid,path=import_raw(root)
    first,_=sanitize_external(registry,rawid)
    identity=first["sanitized_dataset_id"]
    # Model a failure between atomic filesystem publication and catalog commit.
    with registry.catalog.connection() as con:
        for table in ("market_data_lineage","dataset_provenance_seals","provenance","datasets"):
            con.execute(f"DELETE FROM {table} WHERE dataset_id=?",(identity,))
    second,_=sanitize_external(registry,rawid)
    assert first==second
    registry.verify_dataset(identity)


def test_sanitation_reread_detects_tampered_audit(root):
    registry,gateway,rawid,path=import_raw(root)
    m,_=sanitize_external(registry,rawid)
    audit=root/m["artifact_directory"]/"audit.parquet"
    audit.write_bytes(b"corrupt test evidence")
    with pytest.raises(ValueError,match="checksum"):sanitize_external(registry,rawid)
