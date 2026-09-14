"""Integration checks against the actual sanitized Discovery dataset, when provisioned."""
from pathlib import Path
from datetime import timedelta
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from sandbox.research.registry import ResearchRegistry
from sandbox.partition.service import PartitionService
from sandbox.partition.models import AccessContext,AccessOperation
from sandbox.market_state.universal import UniversalFeatureEngine
from sandbox.market_state.universal.continuity import QualityContext
from sandbox.market_data.sanitation import frame_hash
from sandbox.market_data.storage import lineage_record


@pytest.fixture(scope="module")
def real():
    if not Path("data/catalog.sqlite3").exists():pytest.skip("real catalog unavailable")
    registry=ResearchRegistry(Path("data/catalog.sqlite3"))
    with registry.catalog.connection() as con:
        p=con.execute("SELECT partition_id,source_dataset_id FROM data_partitions WHERE program_id='AUTONOMOUS_RD_DATA' AND role='DISCOVERY' AND timeframe='M15'").fetchone()
    if not p:pytest.skip("real sanitized Discovery partition not provisioned")
    service=PartitionService(registry)
    source=service.access(p[0],AccessContext.DISCOVERY_CONTEXT,AccessOperation.FEATURE_ANALYSIS,actor="real-data-verification-tests")
    manifest=lineage_record(registry,p[1])
    matches=[]
    for path in Path("data/derived/features/EURUSD/M15").glob("*/metadata.json"):
        m=json.loads(path.read_text())
        if m["source_partition_fingerprint"]==service.get_partition(p[0]).partition_fingerprint:
            matches.append(m)
    assert matches,"provisioned Discovery dataset must have its feature artifact"
    artifact=matches[-1]
    feature=pd.read_parquet(artifact["artifact_path"])
    return service,source,manifest,artifact,feature


def test_real_exclusions_raw_status_and_complete_row_alignment(real):
    service,source,m,artifact,feature=real
    with service.registry.catalog.connection() as con:
        status=con.execute("SELECT status FROM external_imports WHERE dataset_id=?",(m["parent_raw_dataset_id"],)).fetchone()[0]
    assert status=="QUARANTINED"
    audit=pd.read_parquet(Path(m["artifact_directory"])/"audit.parquet")
    excluded=audit[~audit.retained]
    assert len(excluded)==10 and excluded.classification.eq("INVALID_OHLC").all()
    assert not source.timestamp_utc.isin(excluded.timestamp_utc).any()
    assert len(source)==len(feature)
    assert ((source.timestamp_utc+timedelta(minutes=15)).reset_index(drop=True)==feature.timestamp.reset_index(drop=True)).all()


def test_real_future_mutation_prefix_and_determinism(real):
    source=real[1]
    # Multiple real segments include their actual gaps; never fabricate initial data.
    f=source.iloc[:2500].copy().reset_index(drop=True)
    engine=UniversalFeatureEngine();a=engine.compute(f,symbol="EURUSD",quality_context=QualityContext())
    for cutoff in (120,1100,2100):
        changed=f.copy()
        changed.loc[cutoff+1:,["open","high","low","close"]]=[2.,3.,.5,1.]
        b=engine.compute(changed,symbol="EURUSD",quality_context=QualityContext())
        assert a[:cutoff+1]==b[:cutoff+1]
        prefix=engine.compute(f.iloc[:cutoff+1],symbol="EURUSD",quality_context=QualityContext())
        assert a[:cutoff+1]==prefix
    assert [r.model_dump_json() for r in a]==[r.model_dump_json() for r in engine.compute(f,symbol="EURUSD",quality_context=QualityContext())]


def test_real_all_discontinuity_starts_reset_state(real):
    source,feature=real[1],real[4]
    first=source.continuity_segment_id.ne(source.continuity_segment_id.shift())
    starts=feature.loc[first.to_numpy()]
    assert len(starts)>1
    assert not starts.feature_ready.any()
    for col in ("atr_14","prior_high_12","prior_low_12","displacement_4_atr","completed_h1_direction","completed_h3_direction"):
        assert starts[col].isna().all()


def test_real_context_intraday_boundaries_weekend_and_dst(real):
    source,features=real[1],real[4]
    ready=features[features.feature_ready]
    selected=[ready.iloc[0],ready[ready.timestamp.dt.minute.eq(0)].iloc[0],
              ready[ready.timestamp.dt.minute.eq(0)&ready.timestamp.dt.hour.mod(3).eq(0)].iloc[0]]
    march=features[(features.timestamp>="2024-03-08")&(features.timestamp<"2024-03-13")]
    assert len(march)
    selected.extend([march.iloc[0],march.iloc[-1]])
    starts=features[features.continuity_segment_id.ne(features.continuity_segment_id.shift())]
    weekend=starts[starts.timestamp.dt.weekday.isin([6,0])]
    assert len(weekend);selected.append(weekend.iloc[1])
    for row in selected:
        current=source[(source.continuity_segment_id==row.continuity_segment_id)&
                       (source.timestamp_utc+timedelta(minutes=15)<=row.timestamp)]
        for hours,name in ((1,"completed_h1_direction"),(3,"completed_h3_direction")):
            # Independent arithmetic bucket check, not feature-engine context helpers.
            candidates=[]
            for start,group in current.groupby(current.timestamp_utc.dt.floor(f"{hours}h")):
                end=start+timedelta(hours=hours)
                expected=pd.date_range(start,periods=hours*4,freq="15min")
                if len(group)==hours*4 and list(group.timestamp_utc)==list(expected) and end<=row.timestamp:
                    candidates.append((end,int(group.close.iloc[-1]>group.open.iloc[0])-int(group.close.iloc[-1]<group.open.iloc[0])))
            value=getattr(row,name)
            if candidates:
                assert value==candidates[-1][1] and candidates[-1][0]<=row.timestamp
            else:assert pd.isna(value)


def test_real_feature_artifact_logical_hash_and_holdout_protection(real):
    service,source,m,artifact,feature=real
    assert frame_hash(feature)==artifact["canonical_content_sha256"]
    assert int(feature.feature_ready.sum())==artifact["feature_ready_row_count"]>5
    assert feature.source_dataset_id.eq(m["sanitized_dataset_id"]).all()
    with service.registry.catalog.connection() as con:
        held=con.execute("SELECT partition_id FROM data_partitions WHERE program_id='AUTONOMOUS_RD_DATA' AND role='VALIDATION'").fetchone()[0]
    with pytest.raises(Exception,match="ACCESS_DENIED"):
        service.access(held,AccessContext.DISCOVERY_CONTEXT,AccessOperation.FEATURE_ANALYSIS,actor="holdout-denial-test")
