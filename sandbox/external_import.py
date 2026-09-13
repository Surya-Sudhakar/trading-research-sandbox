from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from sandbox import EXTERNAL_IMPORTER_VERSION, SCHEMA_VERSION
from sandbox.catalog import Catalog
from sandbox.integrity import audit
from sandbox.provenance import Provenance, sha256_file
from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.storage import COLUMNS, safe_segment


class ImportError(ValueError):
    def __init__(self,message:str,diagnostics:dict|None=None):super().__init__(message);self.diagnostics=diagnostics or {}
class ImportStatus(StrEnum): STAGED="STAGED"; VALIDATING="VALIDATING"; CERTIFIED="CERTIFIED"; QUARANTINED="QUARANTINED"; REJECTED="REJECTED"
class Resolution(StrEnum): TICK="TICK"; M1="M1"; M15="M15"
class PriceType(StrEnum): BID="BID"; ASK="ASK"; MID="MID"; BID_ASK="BID_ASK"; UNKNOWN="UNKNOWN"
class TimestampFormat(StrEnum): ISO8601="ISO8601"; UNIX_MILLISECONDS="UNIX_MILLISECONDS"


@dataclass(frozen=True)
class ImportSpec:
    files: tuple[Path, ...]
    provider: str
    source_symbol: str
    canonical_symbol: str
    resolution: Resolution
    price_type: PriceType
    source_timezone: str
    columns: dict[str, str]
    target_execution_broker: str | None = None
    derive_mid: bool = False
    interpretation_reason: str | None = None
    timestamp_format: TimestampFormat = TimestampFormat.ISO8601


def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _safe(value: str, label: str) -> str:
    if not value or safe_segment(value) != value or value in {".", ".."}: raise ImportError(f"unsafe {label}")
    return value
def _format(path: Path) -> str:
    name=path.name.lower()
    if name.endswith(".csv.gz"): return "CSV.GZ"
    if name.endswith(".csv"): return "CSV"
    if name.endswith(".parquet"): return "PARQUET"
    raise ImportError(f"unsupported input format: {path.name}")
def _chunks(path: Path, columns: list[str], size: int=200_000) -> Iterator[pd.DataFrame]:
    fmt=_format(path)
    if fmt in {"CSV","CSV.GZ"}:
        yield from pd.read_csv(path, usecols=columns, chunksize=size)
    else:
        pf=pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=size, columns=columns): yield batch.to_pandas()


class ExternalImportGateway:
    """Local-file-only Stage 1 import gateway. It never accesses a broker or network."""
    def __init__(self, catalog: Catalog, data_dir: Path):
        self.catalog=catalog; self.data_dir=data_dir
        self.raw_root=data_dir/"external_raw"; self.stage_root=data_dir/"import_staging"; self.canonical_root=data_dir/"derived"/"external"
        catalog.initialize()

    def _existing(self, fingerprint: str):
        with self.catalog.connection() as con: row=con.execute("SELECT * FROM external_imports WHERE scientific_fingerprint=?",(fingerprint,)).fetchone()
        return dict(row) if row else None

    def import_files(self, spec: ImportSpec) -> dict:
        started=time.monotonic()
        if not spec.files: raise ImportError("at least one component file is required")
        provider=_safe(spec.provider,"provider"); source_symbol=_safe(spec.source_symbol,"source symbol"); canonical_symbol=_safe(spec.canonical_symbol,"canonical symbol")
        if not spec.source_timezone or spec.source_timezone.upper()=="UNKNOWN": raise ImportError("source timezone must be explicitly declared")
        try: pd.Timestamp("2024-01-01").tz_localize(spec.source_timezone)
        except Exception as exc: raise ImportError(f"invalid source timezone: {spec.source_timezone}") from exc
        paths=tuple(sorted((Path(x).resolve() for x in spec.files),key=lambda p:str(p).casefold()))
        if len(set(paths)) != len(paths): raise ImportError("duplicate component path")
        components=[]
        for p in paths:
            if not p.is_file(): raise ImportError(f"source file not found: {p}")
            components.append({"original_path":str(p),"original_filename":p.name,"sha256":sha256_file(p),"size_bytes":p.stat().st_size,"format":_format(p)})
        identity={"component_sha256s":[x["sha256"] for x in components],"provider":provider,"source_symbol":source_symbol,"canonical_symbol":canonical_symbol,"resolution":spec.resolution.value,"price_type":spec.price_type.value,"source_timezone":spec.source_timezone,"timestamp_format":spec.timestamp_format.value,"normalization_method":"UTC_CANONICAL_V1","importer_version":EXTERNAL_IMPORTER_VERSION,"derive_mid":spec.derive_mid,"column_mapping":spec.columns}
        fingerprint=sha256_canonical(identity); existing=self._existing(fingerprint)
        if existing: return {"outcome":"DUPLICATE_DATASET","existing_dataset_id":existing["dataset_id"],"import_id":existing["import_id"],"status":existing["status"]}
        hashes=tuple(x["sha256"] for x in components)
        with self.catalog.connection() as con:
            prior=con.execute("SELECT manifest_json FROM external_imports").fetchall()
        alternate=[json.loads(x[0]) for x in prior if tuple(json.loads(x[0]).get("component_sha256s",[]))==hashes]
        if alternate and not spec.interpretation_reason: raise ImportError("MULTIPLE_INTERPRETATIONS requires explicit interpretation_reason")
        import_id="IMP-"+uuid4().hex; dataset_id="EXT-"+fingerprint[:32]; work=self.stage_root/import_id
        work.mkdir(parents=True,exist_ok=False); status=ImportStatus.VALIDATING
        audit_data={"rows_read":0,"rows_accepted":0,"rows_rejected":0,"duplicate_events":0,"ordering_issues":0,"ohlc_violations":0,"crossed_quotes":0,"expected_gaps":0,"suspicious_gaps":0,"component_overlap":0,"timezone_conversion":f"{spec.source_timezone} -> UTC","issues":[]}
        promoted_raw=None;promoted_canonical=None;registered=False
        try:
            preserved=[]
            for i,(p,c) in enumerate(zip(paths,components)):
                target=work/f"{i:04d}-{safe_segment(p.name)}"; shutil.copyfile(p,target)
                if sha256_file(target)!=c["sha256"] or sha256_file(p)!=c["sha256"]: raise ImportError("source changed during import")
                preserved.append(target)
            canonical=work/"canonical.parquet"; summary=self._normalize(preserved,spec,audit_data,canonical)
            if not summary["row_count"]: raise ImportError("no valid rows")
            serious=[x for x in audit_data["issues"] if x.get("category")=="confirmed_data_error"]
            status=ImportStatus.QUARANTINED if serious else ImportStatus.STAGED
            final_raw=self.raw_root/provider/dataset_id; final_canonical=self.canonical_root/provider/canonical_symbol/spec.resolution.value/f"{dataset_id}.parquet"
            final_raw.parent.mkdir(parents=True,exist_ok=True); final_canonical.parent.mkdir(parents=True,exist_ok=True)
            if final_raw.exists() or final_canonical.exists(): raise ImportError("artifact target collision")
            os.replace(work,final_raw);promoted_raw=final_raw;canonical=final_raw/"canonical.parquet"
            shutil.copyfile(canonical,final_canonical);promoted_canonical=final_canonical
            first=summary["first_timestamp"]; last=summary["last_timestamp"]
            manifest={"dataset_id":dataset_id,"import_id":import_id,"data_source_provider":provider,"provider_authentication":"DECLARED_PROVENANCE","source_type":"EXTERNAL_HISTORICAL","target_execution_broker":spec.target_execution_broker,"source_symbol":source_symbol,"canonical_symbol":canonical_symbol,"resolution":spec.resolution.value,"price_type":spec.price_type.value,"source_timezone":spec.source_timezone,"timestamp_format":spec.timestamp_format.value,"canonical_timezone":"UTC","normalization_timezone":"UTC","original_filenames":[x["original_filename"] for x in components],"component_files":[Path(x).name for x in preserved],"component_sha256s":list(hashes),"component_sizes":[x["size_bytes"] for x in components],"row_count":summary["row_count"],"first_timestamp":first,"last_timestamp":last,"integrity_summary":audit_data,"normalization_method":"UTC_CANONICAL_V1","importer_version":EXTERNAL_IMPORTER_VERSION,"canonical_dataset_sha256":sha256_file(final_canonical),"canonical_path":str(final_canonical),"created_at":_now(),"scientific_fingerprint":fingerprint,"multiple_interpretations":bool(alternate),"interpretation_reason":spec.interpretation_reason,"column_mapping":spec.columns,"derive_mid":spec.derive_mid,"spread_units":"PRICE_DISTANCE" if spec.resolution==Resolution.TICK and spec.price_type==PriceType.BID_ASK else None,"initial_classification":"UNASSIGNED_RESEARCH_DATA","processing":{"file_size_bytes":sum(x["size_bytes"] for x in components),"duration_seconds":time.monotonic()-started,"approach":"bounded-memory chunked parsing and incremental Parquet writing","input_chunk_rows":200000}}
            mf=sha256_canonical(manifest); now=_now()
            (final_raw/"manifest.json").write_text(json.dumps({**manifest,"manifest_fingerprint":mf},indent=2,sort_keys=True),encoding="utf-8")
            (final_raw/"audit.json").write_text(json.dumps(audit_data,indent=2,sort_keys=True),encoding="utf-8")
            with self.catalog.connection() as con:
                con.execute("INSERT INTO external_imports VALUES (?,?,?,?,?,?,?,?,?)",(import_id,dataset_id,fingerprint,status.value,canonical_json(manifest),mf,canonical_json(audit_data),now,now))
                for i,(c,p) in enumerate(zip(components,preserved)): con.execute("INSERT INTO external_import_components VALUES (?,?,?,?,?,?)",(import_id,i,c["original_path"],str(final_raw/p.name),c["sha256"],c["size_bytes"]))
            registered=True
            return {"outcome":"MULTIPLE_INTERPRETATIONS" if alternate else "IMPORTED","import_id":import_id,"dataset_id":dataset_id,"status":status.value,"manifest_fingerprint":mf,"audit":audit_data}
        except Exception as exc:
            if work.exists(): shutil.rmtree(work,ignore_errors=True)
            if not registered and promoted_canonical and promoted_canonical.exists():promoted_canonical.unlink()
            if not registered and promoted_raw and promoted_raw.exists():shutil.rmtree(promoted_raw,ignore_errors=True)
            if isinstance(exc,ImportError) and not exc.diagnostics:exc.diagnostics=audit_data
            raise

    def _normalize(self, paths: list[Path], spec: ImportSpec, result: dict, target: Path) -> dict:
        ohlc=spec.resolution in {Resolution.M1,Resolution.M15}
        required=["timestamp","open","high","low","close"] if ohlc else ["timestamp","bid","ask"]
        if any(x not in spec.columns for x in required): raise ImportError("explicit column mapping missing: "+",".join(x for x in required if x not in spec.columns))
        writer=None;last=None;seen_m1=set();seen_events=set();row_count=0;first_timestamp=None;last_timestamp=None
        for component_index,path in enumerate(paths):
            before_component_m1=set(seen_m1);before_component_events=set(seen_events)
            for raw in _chunks(path,[spec.columns[x] for x in spec.columns]):
                result["rows_read"]+=len(raw); f=raw.rename(columns={v:k for k,v in spec.columns.items()})
                try:
                    if spec.timestamp_format==TimestampFormat.UNIX_MILLISECONDS:
                        values=pd.to_numeric(f["timestamp"],errors="raise")
                        if not values.map(math.isfinite).all() or not (values%1==0).all():raise ValueError("Unix-millisecond timestamps must be finite integers")
                        ts=pd.to_datetime(values.astype("int64"),unit="ms",origin="unix",utc=True,errors="raise")
                    elif spec.timestamp_format==TimestampFormat.ISO8601:ts=pd.to_datetime(f["timestamp"],errors="raise",format="mixed")
                    else:raise ValueError("unsupported timestamp representation")
                except Exception: result["issues"].append({"category":"confirmed_data_error","code":"invalid_timestamp","detail":f"{len(f)} rows"}); continue
                if getattr(ts.dt,"tz",None) is None:
                    try: ts=ts.dt.tz_localize(spec.source_timezone,ambiguous="raise",nonexistent="raise")
                    except Exception as exc: result["issues"].append({"category":"confirmed_data_error","code":"timezone_ambiguity","detail":str(exc)}); continue
                else: ts=ts.dt.tz_convert(spec.source_timezone)
                f["timestamp_utc"]=ts.dt.tz_convert("UTC")
                misaligned=pd.Series(False,index=f.index)
                if spec.resolution==Resolution.M15:
                    misaligned=(f.timestamp_utc.dt.minute%15!=0)|(f.timestamp_utc.dt.second!=0)|(f.timestamp_utc.dt.microsecond!=0)
                    if misaligned.any():result["issues"].append({"category":"confirmed_data_error","code":"misaligned_m15_timestamp","detail":f"{int(misaligned.sum())} rows"})
                if last is not None and len(f) and f.timestamp_utc.iloc[0] < last: result["ordering_issues"]+=1
                if not f.timestamp_utc.is_monotonic_increasing: result["ordering_issues"]+=1
                if len(f): last=f.timestamp_utc.iloc[-1]
                nums=["open","high","low","close"] if ohlc else ["bid","ask"]
                for c in nums: f[c]=pd.to_numeric(f[c],errors="coerce")
                invalid=f[nums].isna().any(axis=1) | (~f[nums].map(math.isfinite)).any(axis=1) | (f[nums]<=0).any(axis=1)
                if invalid.any(): result["issues"].append({"category":"confirmed_data_error","code":"invalid_numeric_price","detail":f"{int(invalid.sum())} rows"})
                if ohlc:
                    bad=(f.high<f[["open","close"]].max(axis=1))|(f.low>f[["open","close"]].min(axis=1))|(f.high<f.low);result["ohlc_violations"]+=int(bad.sum())
                    if bad.any(): result["issues"].append({"category":"confirmed_data_error","code":"invalid_ohlc","detail":f"{int(bad.sum())} rows"})
                    cross=sum(x in before_component_m1 for x in f.timestamp_utc);duplicates=sum(x in seen_m1 for x in f.timestamp_utc); seen_m1.update(f.timestamp_utc); duplicates+=int(f.timestamp_utc.duplicated().sum());result["duplicate_events"]+=duplicates;result["component_overlap"]+=cross
                    if duplicates: result["issues"].append({"category":"confirmed_data_error","code":"duplicate_ohlc_timestamp","detail":f"{duplicates} rows"})
                    for optional in ("tick_volume","spread","real_volume"):
                        f[optional]=pd.to_numeric(f[optional],errors="coerce") if optional in f else 0
                    out=f[["timestamp_utc",*COLUMNS[1:]]]
                else:
                    crossed=f.ask<f.bid;result["crossed_quotes"]+=int(crossed.sum())
                    if crossed.any(): result["issues"].append({"category":"confirmed_data_error","code":"crossed_quote","detail":f"{int(crossed.sum())} rows"})
                    keys=list(zip(f.timestamp_utc,f.bid,f.ask));cross=sum(k in before_component_events for k in keys);result["component_overlap"]+=cross; result["duplicate_events"]+=sum(k in seen_events for k in keys);seen_events.update(keys)
                    for optional in ("bid_volume","ask_volume"):
                        if optional in f:f[optional]=pd.to_numeric(f[optional],errors="coerce")
                    out=f[["timestamp_utc","bid","ask",*([x for x in ("bid_volume","ask_volume") if x in f])]]
                rejected=(invalid|bad|misaligned) if ohlc else (invalid|crossed);result["rows_rejected"]+=int(rejected.sum()); result["rows_accepted"]+=int((~rejected).sum())
                if len(out):
                    current_first=out.timestamp_utc.iloc[0];current_last=out.timestamp_utc.iloc[-1];first_timestamp=first_timestamp or current_first;last_timestamp=current_last;row_count+=len(out)
                    table=pa.Table.from_pandas(out,preserve_index=False)
                    if writer is None:writer=pq.ParquetWriter(target,table.schema,compression="snappy")
                    writer.write_table(table)
        if writer is not None:writer.close()
        if result["ordering_issues"]: result["issues"].append({"category":"confirmed_data_error","code":"out_of_order","detail":str(result["ordering_issues"])})
        if result["component_overlap"]:result["issues"].append({"category":"confirmed_data_error","code":"component_overlap","detail":str(result["component_overlap"])})
        if ohlc and target.exists():
            expected_seconds={Resolution.M1:60,Resolution.M15:900}[spec.resolution]
            previous=None
            for batch in pq.ParquetFile(target).iter_batches(columns=["timestamp_utc"],batch_size=200000):
                for current in pd.to_datetime(batch.column(0).to_pandas(),utc=True):
                    current=pd.Timestamp(current).as_unit("ns")
                    gap_seconds=(current.to_pydatetime()-previous.to_pydatetime()).total_seconds() if previous is not None else 0
                    if previous is not None and gap_seconds>expected_seconds:
                        gap=current-previous
                        if previous.weekday()==4 and current.weekday()==6 and previous.hour>=20 and 20<=current.hour<=23:result["expected_gaps"]+=1
                        else:result["suspicious_gaps"]+=1
                    previous=current
        return {"row_count":row_count,"first_timestamp":first_timestamp.isoformat() if first_timestamp is not None else None,"last_timestamp":last_timestamp.isoformat() if last_timestamp is not None else None}

    def derive_m1(self, dataset_id: str, price: PriceType) -> pd.DataFrame:
        record=self.inspect_dataset(dataset_id); m=record["manifest"]
        if m["resolution"]!="TICK": raise ImportError("tick derivation requires TICK dataset")
        if price==PriceType.MID and not m.get("derive_mid"): raise ImportError("MID derivation was not explicitly requested")
        if price not in {PriceType.BID,PriceType.ASK,PriceType.MID}: raise ImportError("derive price must be BID, ASK, or MID")
        ticks=pd.read_parquet(m["canonical_path"]); ticks["timestamp_utc"]=pd.to_datetime(ticks.timestamp_utc,utc=True)
        values=(ticks.bid+ticks.ask)/2 if price==PriceType.MID else ticks[price.value.lower()]
        work=pd.DataFrame({"timestamp_utc":ticks.timestamp_utc,"price":values,"spread":ticks.ask-ticks.bid}); work["minute"]=work.timestamp_utc.dt.floor("min")
        bars=work.groupby("minute",sort=True).agg(open=("price","first"),high=("price","max"),low=("price","min"),close=("price","last"),tick_volume=("price","size"),spread_open=("spread","first"),spread_close=("spread","last"),spread_min=("spread","min"),spread_max=("spread","max"),spread_mean=("spread","mean")).reset_index().rename(columns={"minute":"timestamp_utc"})
        bars["spread"]=0;bars["real_volume"]=0
        return bars[[*COLUMNS,"spread_open","spread_close","spread_min","spread_max","spread_mean"]]

    def inspect_dataset(self,dataset_id: str) -> dict:
        with self.catalog.connection() as con: row=con.execute("SELECT * FROM external_imports WHERE dataset_id=? OR import_id=?",(dataset_id,dataset_id)).fetchone()
        if not row: raise ImportError("external import not found")
        value=dict(row);value["manifest"]=json.loads(value.pop("manifest_json"));value["audit"]=json.loads(value.pop("audit_json"))
        if sha256_canonical(value["manifest"])!=value["manifest_fingerprint"]:raise ImportError("external manifest fingerprint mismatch")
        return value

    def list_imports(self) -> list[dict]:
        with self.catalog.connection() as con:return [dict(x) for x in con.execute("SELECT import_id,dataset_id,status,created_at,updated_at FROM external_imports ORDER BY created_at")]

    def certify(self,dataset_id: str) -> dict:
        record=self.inspect_dataset(dataset_id)
        if record["status"]==ImportStatus.CERTIFIED.value:return {"dataset_id":record["dataset_id"],"status":"CERTIFIED"}
        if record["status"]!=ImportStatus.STAGED.value: raise ImportError(f"cannot certify dataset in {record['status']} state")
        m=record["manifest"]
        with self.catalog.connection() as con: components=con.execute("SELECT * FROM external_import_components WHERE import_id=? ORDER BY ordinal",(record["import_id"],)).fetchall()
        for c in components:
            if sha256_file(Path(c["original_path"]))!=c["sha256"] or sha256_file(Path(c["preserved_path"]))!=c["sha256"]: raise ImportError("component checksum mismatch; certification blocked")
        canonical=Path(m["canonical_path"])
        if sha256_file(canonical)!=m["canonical_dataset_sha256"]: raise ImportError("canonical checksum mismatch; certification blocked")
        prov=Provenance(record["dataset_id"],m["data_source_provider"],"EXTERNAL_HISTORICAL",m["canonical_symbol"],m["resolution"],m["first_timestamp"],m["last_timestamp"],m["created_at"],m["row_count"],SCHEMA_VERSION,m["canonical_dataset_sha256"],str(canonical))
        now=_now()
        with self.catalog.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(prov.dataset_id,prov.broker,prov.server,prov.symbol,prov.timeframe,prov.start,prov.end,prov.row_count,prov.path,prov.checksum,prov.schema_version,prov.retrieved_at))
            p=json.dumps(prov.to_dict(),sort_keys=True);import hashlib
            con.execute("INSERT INTO provenance VALUES (?,?)",(prov.dataset_id,p));con.execute("INSERT INTO dataset_provenance_seals(dataset_id,provenance_fingerprint) VALUES (?,?)",(prov.dataset_id,hashlib.sha256(p.encode()).hexdigest()))
            con.execute("UPDATE external_imports SET status='CERTIFIED',updated_at=? WHERE dataset_id=?",(now,dataset_id))
            con.execute("INSERT OR IGNORE INTO dataset_classifications VALUES (?,?,?,?)",(dataset_id,"UNASSIGNED_RESEARCH_DATA","External import; roles require explicit Stage 6 partitioning",now))
        return {"dataset_id":dataset_id,"status":"CERTIFIED","checksum":prov.checksum,"classification":"UNASSIGNED_RESEARCH_DATA"}
