from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sandbox.provenance import sha256_file
from sandbox.research.canonical import canonical_json,sha256_canonical
from .models import MarketStateSnapshot,MarketStateSnapshotLedger,SnapshotLink


@dataclass(frozen=True)
class StoredSnapshotLedger:
    snapshot_path:Path;link_path:Path;manifest_path:Path;snapshot_checksum:str;link_checksum:str;ledger_fingerprint:str


def create_snapshot_ledger(snapshots,links=()):
    snapshots=tuple(sorted(snapshots,key=lambda x:(x.information_cutoff_timestamp,x.snapshot_id)))
    links=tuple(sorted(links,key=lambda x:(x.snapshot_id,x.signal_id or "",x.setup_id or "",x.trade_id or "")))
    link_rows=[{"snapshot_id":x.snapshot_id,"snapshot_phase":x.snapshot_phase,"setup_id":x.setup_id,"signal_id":x.signal_id,"trade_id":x.trade_id} for x in links]
    fingerprint=sha256_canonical({"snapshots":[x.canonical_dict() for x in snapshots],"links":link_rows})
    return MarketStateSnapshotLedger(snapshots,links,fingerprint)


def store_snapshot_ledger(ledger:MarketStateSnapshotLedger,root:Path,ledger_id:str):
    target=root/ledger_id
    if target.exists():raise FileExistsError("market-state snapshot ledger is immutable")
    target.mkdir(parents=True)
    snapshot_path=target/"snapshots.parquet";link_path=target/"links.parquet"
    rows=[]
    for snapshot in ledger.snapshots:
        row=snapshot.canonical_dict();row["values_json"]=canonical_json(row.pop("values"));row["validity_flags_json"]=canonical_json(row.pop("validity_flags"));row["missing_features_json"]=canonical_json(row.pop("missing_features"));row["data_quality_flags_json"]=canonical_json(row.pop("data_quality_flags"));rows.append(row)
    pd.DataFrame(rows).to_parquet(snapshot_path,index=False)
    pd.DataFrame([{"snapshot_id":x.snapshot_id,"snapshot_phase":x.snapshot_phase.value,"setup_id":x.setup_id,"signal_id":x.signal_id,"trade_id":x.trade_id} for x in ledger.links]).to_parquet(link_path,index=False)
    manifest={"ledger_fingerprint":ledger.ledger_fingerprint,"snapshot_checksum":sha256_file(snapshot_path),"link_checksum":sha256_file(link_path),"snapshot_count":len(ledger.snapshots),"link_count":len(ledger.links)}
    manifest_path=target/"manifest.json";manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding="utf-8")
    return StoredSnapshotLedger(snapshot_path,link_path,manifest_path,manifest["snapshot_checksum"],manifest["link_checksum"],ledger.ledger_fingerprint)
