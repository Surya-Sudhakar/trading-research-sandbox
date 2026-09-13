from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from sandbox.provenance import sha256_file
from .models import SignalLedger
@dataclass(frozen=True)
class StoredSignalLedger:path:Path;checksum:str;ledger_fingerprint:str;row_count:int
def store_signal_ledger(ledger:SignalLedger,root:Path,run_id:str)->StoredSignalLedger:
    import pandas as pd
    target=root/run_id/"signals.parquet"
    if target.exists():raise FileExistsError("signal ledger is immutable")
    target.parent.mkdir(parents=True,exist_ok=True);rows=[]
    for record in ledger.records:
        row=record.model_dump(mode="json");row["metadata_json"]=json.dumps(row.pop("metadata"),sort_keys=True,separators=(",",":"));rows.append(row)
    pd.DataFrame(rows).to_parquet(target,index=False)
    meta={"ledger_fingerprint":ledger.ledger_fingerprint,"checksum":sha256_file(target),"row_count":len(ledger.records)};target.with_suffix(".manifest.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
    return StoredSignalLedger(target,meta["checksum"],ledger.ledger_fingerprint,len(ledger.records))
