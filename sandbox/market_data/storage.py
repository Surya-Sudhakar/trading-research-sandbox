from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import os
from uuid import uuid4
import pandas as pd
from sandbox.external_import import ExternalImportGateway
from sandbox.provenance import Provenance, sha256_file
from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.market_state.universal.dataset import validate_bars
from .providers import ExternalCSVAdapter
from .models import ProviderMetadata
from .sanitation import sanitize
from .gaps import SessionProfile, ContinuityPolicy

DDL="""CREATE TABLE IF NOT EXISTS market_data_lineage(
 dataset_id TEXT PRIMARY KEY,kind TEXT NOT NULL,parent_dataset_id TEXT,
 manifest_json TEXT NOT NULL,manifest_sha256 TEXT NOT NULL);"""


def validation_view(frame):
    """Legacy integrity validator view only; optional unknowns remain null in storage."""
    result=frame.copy()
    for name in ("tick_volume","spread","real_volume"):
        result[name]=pd.to_numeric(result[name],errors="raise").fillna(0).astype(float) if name in result else 0.
    return result


def lineage_record(registry, dataset_id):
    with registry.catalog.connection() as con:
        con.executescript(DDL)
        row=con.execute("SELECT manifest_json,manifest_sha256 FROM market_data_lineage WHERE dataset_id=?",(dataset_id,)).fetchone()
    if row is None:return None
    manifest=json.loads(row[0])
    if sha256_canonical(manifest)!=row[1]:raise ValueError("lineage manifest integrity failed")
    return manifest


def sanitize_external(registry, raw_dataset_id, profile=SessionProfile(), continuity=ContinuityPolicy()):
    gateway=ExternalImportGateway(registry.catalog,registry.project_root/"data")
    parent=gateway.inspect_dataset(raw_dataset_id)
    m=parent["manifest"]
    if m["resolution"] not in {"M1","M15"}:raise ValueError("OHLC source required")
    with registry.catalog.connection() as con:
        components=con.execute("SELECT preserved_path,sha256 FROM external_import_components WHERE import_id=? ORDER BY ordinal",(parent["import_id"],)).fetchall()
    if len(components)!=1:raise ValueError("V1 sanitation requires one source file; component reconciliation required")
    source=Path(components[0][0])
    if not source.is_absolute():source=registry.project_root/source
    if sha256_file(source)!=components[0][1] or components[0][1]!=m["component_sha256s"][0]:
        raise ValueError("immutable raw source checksum mismatch")
    adapter=ExternalCSVAdapter(source,m["column_mapping"],m.get("timestamp_format","ISO8601"),m["source_timezone"])
    metadata=ProviderMetadata(m["data_source_provider"],m["canonical_symbol"],m["source_symbol"],
                              m["resolution"],m["price_type"],raw_dataset_id)
    result=sanitize(adapter.read(),metadata,profile,continuity)
    bars,audit,gaps,manifest=result
    if bars.empty:raise ValueError("no valid retained bars")
    validate_bars(validation_view(bars),metadata.timeframe)
    manifest.update(source_sha256=components[0][1],raw_status=parent["status"],kind="SANITIZED",
                    raw_import_timestamp=m["created_at"],original_source_identifier=m["original_filenames"][0],
                    timestamp_convention=m.get("timestamp_format","ISO8601")+" -> UTC",
                    parent_manifest_fingerprint=parent["manifest_fingerprint"])
    identity=sha256_canonical(manifest)
    dataset_id="SAN-"+identity[:32]
    existing=lineage_record(registry,dataset_id)
    if existing:
        registry.verify_dataset(dataset_id)
        if json.loads((registry.project_root/Path(existing["artifact_directory"])/"manifest.json").read_text())!=existing:
            raise ValueError("sanitation manifest sidecar mismatch")
        if existing["sanitized_content_fingerprint"]!=manifest["sanitized_content_fingerprint"]:
            raise ValueError("sanitized identity mismatch")
        for filename,key in (("audit.parquet","audit_sha256"),("gaps.parquet","gaps_sha256")):
            if sha256_file(registry.project_root/Path(existing["artifact_directory"])/filename)!=existing[key]:
                raise ValueError("sanitation evidence checksum mismatch")
        return existing,result
    relative=Path("data/derived/sanitized")/metadata.provider/metadata.symbol/metadata.timeframe/dataset_id
    target=registry.project_root/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.with_name(dataset_id+".staging-"+uuid4().hex);staging.mkdir()
    bars.to_parquet(staging/"bars.parquet",index=False)
    audit.to_parquet(staging/"audit.parquet",index=False)
    gaps.to_parquet(staging/"gaps.parquet",index=False)
    created=datetime.now(timezone.utc).isoformat()
    manifest.update(sanitized_dataset_id=dataset_id,created_at=created,artifact_directory=relative.as_posix(),
                    sanitized_parquet_sha256=sha256_file(staging/"bars.parquet"),
                    audit_sha256=sha256_file(staging/"audit.parquet"),gaps_sha256=sha256_file(staging/"gaps.parquet"))
    (staging/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if target.exists():
        # A complete unregistered artifact from an interrupted run can be reused.
        prior=json.loads((target/"manifest.json").read_text())
        runtime_fields={"sanitized_dataset_id","created_at","artifact_directory","sanitized_parquet_sha256","audit_sha256","gaps_sha256"}
        if sha256_canonical({k:v for k,v in prior.items() if k not in runtime_fields})!=identity:
            raise ValueError("artifact collision")
        for filename,key in (("bars.parquet","sanitized_parquet_sha256"),("audit.parquet","audit_sha256"),("gaps.parquet","gaps_sha256")):
            if sha256_file(target/filename)!=prior[key]:raise ValueError("unregistered artifact checksum mismatch")
        manifest=prior
        for child in staging.iterdir():child.unlink()
        staging.rmdir()
    else:os.replace(staging,target)
    prov=Provenance(dataset_id,metadata.provider,"SANITIZED_RESEARCH",metadata.symbol,metadata.timeframe,
                    manifest["first_timestamp"],manifest["last_timestamp"],manifest["created_at"],len(bars),"1",
                    manifest["sanitized_parquet_sha256"],(relative/"bars.parquet").as_posix())
    # Dataset registration and lineage are one transaction; original import status is untouched.
    with registry.catalog.connection() as con:
        con.executescript(DDL)
        con.execute("BEGIN IMMEDIATE")
        con.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (prov.dataset_id,prov.broker,prov.server,prov.symbol,prov.timeframe,prov.start,prov.end,
                     prov.row_count,prov.path,prov.checksum,prov.schema_version,prov.retrieved_at))
        record=json.dumps(prov.to_dict(),sort_keys=True)
        import hashlib
        con.execute("INSERT INTO provenance VALUES (?,?)",(dataset_id,record))
        con.execute("INSERT INTO dataset_provenance_seals(dataset_id,provenance_fingerprint) VALUES (?,?)",(dataset_id,hashlib.sha256(record.encode()).hexdigest()))
        con.execute("INSERT INTO market_data_lineage VALUES (?,?,?,?,?)",(dataset_id,"SANITIZED",raw_dataset_id,
                    canonical_json(manifest),sha256_canonical(manifest)))
    if sha256_file(source)!=components[0][1] or gateway.inspect_dataset(raw_dataset_id)["status"]!=parent["status"]:
        raise ValueError("raw parent changed during sanitation")
    registry.append_journal("SANITIZED_DATASET_CREATED","DATA_INGESTION",
                            "Immutable derivative registered; raw status retained",
                            metadata=dict(dataset_id=dataset_id,parent_raw_dataset_id=raw_dataset_id,
                                          fingerprint=manifest["sanitized_content_fingerprint"]))
    return manifest,result
