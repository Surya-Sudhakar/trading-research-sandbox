"""Strict preflight and provenance-bound Discovery feature export.

This module cannot import, certify, repair, or reclassify a dataset.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sandbox.integrity import audit, _weekend_closure
from sandbox.market_state.normalization import timeframe_minutes
from sandbox.partition.models import AccessContext, AccessOperation
from sandbox.partition.service import PartitionService
from sandbox.provenance import sha256_file
from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.research.errors import ResearchError
from sandbox.research.registry import ResearchRegistry
from .engine import UniversalFeatureEngine
from .continuity import QualityContext

ARTIFACT_VERSION = "UNIVERSAL_FEATURE_DATASET_V2"


def validate_bars(frame: pd.DataFrame, timeframe: str) -> dict:
    """Strict values/grid validation; gaps reported, never filled."""
    minutes = timeframe_minutes(timeframe)
    if minutes <= 0 or frame.empty:
        raise ValueError("positive timeframe and nonempty bars required")
    times = [pd.Timestamp(t) for t in frame.timestamp_utc]
    if any(pd.isna(t) or t.tzinfo is None for t in times):
        raise ValueError("valid timezone-aware timestamps required")
    ts = pd.DatetimeIndex(times).tz_convert("UTC")
    if (ts != ts.floor(f"{minutes}min")).any():
        raise ValueError("off-grid bar timestamp")
    if {"provider","quality_status","continuity_segment_id"}.issubset(frame):
        from sandbox.market_data.storage import validation_view
        report = audit(validation_view(frame), expected=timedelta(minutes=minutes))
    else:report = audit(frame, expected=timedelta(minutes=minutes))
    if report.status == "error":
        raise ValueError("strict data validation failed: " + ",".join(
            x.code for x in report.issues if x.category == "confirmed_data_error"))
    gaps = []
    for left, right in zip(ts[:-1], ts[1:]):
        elapsed = (right-left).total_seconds()/60
        if elapsed > minutes:
            gaps.append(dict(previous=left.isoformat(), current=right.isoformat(),
                             missing_slots=int(elapsed/minutes)-1,
                             elapsed_minutes=elapsed,
                             classification="EXPECTED_MARKET_GAP" if _weekend_closure(left, right)
                             else "UNEXPLAINED_DATA_GAP"))
    return dict(row_count=len(frame), earliest=ts[0].isoformat(), latest=ts[-1].isoformat(),
                validation_status="VALID_WITH_GAPS" if gaps else "VALID",
                expected_market_gaps=sum(g["classification"] == "EXPECTED_MARKET_GAP" for g in gaps),
                unexplained_data_gaps=sum(g["classification"] == "UNEXPLAINED_DATA_GAP" for g in gaps),
                missing_grid_slots=sum(g["missing_slots"] for g in gaps),
                large_unexplained_gaps=sum(g["classification"] == "UNEXPLAINED_DATA_GAP"
                                           and g["elapsed_minutes"] >= 1440 for g in gaps),
                gaps=gaps)


def logical_hash(rows) -> str:
    """SHA256 of UTF-8 canonical JSON rows, each followed by LF; excludes metadata."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update((canonical_json(row.model_dump(mode="json"))+"\n").encode("utf-8"))
    return digest.hexdigest()


def export_discovery_features(service: PartitionService, partition_id: str) -> dict:
    manifest = service.get_partition(partition_id)
    if manifest is None:
        raise ResearchError("partition not found")
    # Access authorization occurs before any source data or feature computation.
    frame = service.access(partition_id, AccessContext.DISCOVERY_CONTEXT,
                           AccessOperation.FEATURE_ANALYSIS, actor="universal-feature-export")
    verification = service.verify_partition(partition_id)
    if not verification["checksum_valid"] or not verification["row_count_valid"]:
        raise ResearchError("partition integrity failed")
    source = service.registry.verify_dataset(manifest.source_dataset_id)
    if source["symbol"] != manifest.symbol or source["timeframe"] != manifest.timeframe:
        raise ResearchError("partition/source metadata mismatch")
    source_quality = validate_bars(frame, manifest.timeframe)
    if manifest.timeframe not in {"M1", "M15"}:
        raise ValueError("V1 export requires M1 or M15 source partition")
    lineage = [dict(step="AUTHORIZED_DISCOVERY", dataset_id=manifest.source_dataset_id,
                    partition_id=partition_id, timeframe=manifest.timeframe)]
    if manifest.timeframe == "M1":
        request = SimpleNamespace(metadata=SimpleNamespace(required_timeframes=("M15",)))
        if "continuity_segment_id" in frame:
            parts=[]
            for segment,group in frame.groupby("continuity_segment_id",sort=False):
                part=service._strategy_timeframe_frames(group,request,"M1")["M15"]
                part["continuity_segment_id"]=segment;parts.append(part)
            frame=pd.concat(parts,ignore_index=True)
        else:frame = service._strategy_timeframe_frames(frame, request, "M1")["M15"]
        lineage.append(dict(step="COMPLETE_UTC_BUCKET_AGGREGATION", source_timeframe="M1",
                            target_timeframe="M15", incomplete_buckets="DROPPED"))
    quality = validate_bars(frame, "M15")
    engine = UniversalFeatureEngine()
    quality_context=QualityContext() if "continuity_segment_id" in frame else None
    first = engine.compute(frame, symbol=manifest.symbol,quality_context=quality_context)
    ready = sum(row.feature_ready for row in first)
    if ready < 5:
        raise ResearchError(f"INSUFFICIENT_READY_FEATURE_ROWS: {ready}; need at least 5")
    second = engine.compute(frame, symbol=manifest.symbol,quality_context=quality_context)
    fingerprint = logical_hash(first)
    if first != second or fingerprint != logical_hash(second):
        raise ResearchError("feature reproducibility failed")
    del second
    output=pd.DataFrame([r.model_dump() for r in first])
    output["provider"]=source["broker"]
    output["source_dataset_id"]=manifest.source_dataset_id
    output["continuity_segment_id"]=frame.continuity_segment_id.to_numpy() if quality_context else 0
    from sandbox.market_data.sanitation import frame_hash
    from sandbox.market_data.storage import lineage_record
    parent_lineage=lineage_record(service.registry,manifest.source_dataset_id)
    fingerprint=frame_hash(output)
    lineage.extend(dict(step="COMPLETED_CONTEXT", source_timeframe="M15", target_timeframe=tf,
                        anchor="UTC_FIXED", availability="context_end <= decision_close")
                   for tf in ("H1", "H3"))
    code = service.registry.capture_code_version()
    definitions = [asdict(d) for d in engine.definitions]
    identity = dict(feature_schema_version=ARTIFACT_VERSION, definitions=definitions,
                    source_dataset_checksum=source["checksum"],
                    source_partition_fingerprint=manifest.partition_fingerprint,
                    engine_code_fingerprint=code["revision"], symbol=manifest.symbol,
                    decision_timeframe="M15", analysis_timezone=engine.analysis_timezone,
                    canonical_content_sha256=fingerprint,
                    quality_mode="SEGMENT_LOCAL_V1" if quality_context else "LEGACY_OBSERVED_BARS",
                    source_sanitized_content_fingerprint=parent_lineage.get("sanitized_content_fingerprint") if parent_lineage else None)
    artifact_id = sha256_canonical(identity)
    target = service.registry.project_root/"data"/"derived"/"features"/manifest.symbol/"M15"/artifact_id
    if target.exists():
        metadata = json.loads((target/"metadata.json").read_text(encoding="utf-8"))
        if (metadata["canonical_content_sha256"] != fingerprint
                or sha256_file(target/"features.parquet") != metadata["parquet_sha256"]):
            raise ResearchError("existing feature artifact integrity failed")
        return metadata
    target.mkdir(parents=True, exist_ok=False)
    output.to_parquet(target/"features.parquet", index=False)
    if frame_hash(pd.read_parquet(target/"features.parquet"))!=fingerprint:
        raise ResearchError("feature Parquet logical reread mismatch")
    metadata = dict(**identity, artifact_id=artifact_id, source_dataset_id=manifest.source_dataset_id,
                    provider=source["broker"], source_timeframe=source["timeframe"],
                    source_import_timestamp=source["retrieved_at"],
                    timezone_convention="UTC bar opens; UTC feature decision closes",
                    row_count=len(first), feature_ready_row_count=ready,
                    earliest_timestamp=first[0].timestamp.isoformat(),
                    latest_timestamp=first[-1].timestamp.isoformat(),
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    feature_definitions_version=2, source_validation=source_quality,
                    decision_validation=quality, transformation_history=lineage,
                    parquet_sha256=sha256_file(target/"features.parquet"),
                    artifact_path=(target/"features.parquet").relative_to(service.registry.project_root).as_posix(),
                    reproducibility="two equal feature-row sequences; ordered-column canonical JSONL hash including provenance/segment fields; Parquet logical reread verified")
    (target/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    service.registry.append_journal("FEATURE_DATASET_CREATED", manifest.program_id,
                                   "Provenance-bound Discovery feature artifact created",
                                   metadata=dict(artifact_id=artifact_id, partition_id=partition_id,
                                                 canonical_content_sha256=fingerprint))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition_id")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.sqlite3"))
    args = parser.parse_args()
    service = PartitionService(ResearchRegistry(args.catalog))
    print(json.dumps(export_discovery_features(service, args.partition_id), indent=2))


if __name__ == "__main__":
    main()
