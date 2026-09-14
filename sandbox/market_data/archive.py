"""Read-only incremental M1 feed archive, immutable content-addressed monthly revisions."""
from datetime import datetime,timezone
from pathlib import Path
import json,os
import pandas as pd
from sandbox.provenance import sha256_file
from sandbox.market_state.universal.dataset import validate_bars
from .sanitation import frame_hash
from .gaps import inventory,SessionProfile
from .models import ProviderMetadata


def merge_archive(existing,incoming):
    columns=["timestamp_utc","open","high","low","close","tick_volume","spread","real_volume"]
    combined=pd.concat([existing,incoming],ignore_index=True)
    if combined.empty:return combined
    if any(pd.Timestamp(t).tzinfo is None for t in combined.timestamp_utc):raise ValueError("aware archive timestamps required")
    combined["timestamp_utc"]=pd.to_datetime(combined.timestamp_utc,utc=True)
    if not set(columns[:5]).issubset(combined):raise ValueError("OHLC fields required")
    for name in columns[5:]:
        if name not in combined:combined[name]=float("nan")
    combined=combined[columns]
    duplicates=combined[combined.timestamp_utc.duplicated(keep=False)]
    for _,group in duplicates.groupby("timestamp_utc"):
        if len(group.drop_duplicates())>1:raise ValueError("CONFLICTING_ARCHIVE_DUPLICATE")
    return combined.drop_duplicates("timestamp_utc").sort_values("timestamp_utc",kind="stable").reset_index(drop=True)


class MT5Archive:
    def __init__(self,root,provider="IC_MARKETS",symbol="EURUSD",provider_symbol=None):
        from sandbox.storage import safe_segment
        if any(safe_segment(x)!=x or x in {".",".."} for x in (provider,symbol)):raise ValueError("unsafe archive component")
        self.root=Path(root)/provider/symbol/"M1";self.provider=provider;self.symbol=symbol;self.provider_symbol=provider_symbol or symbol

    def read(self):
        if not self.root.exists():return pd.DataFrame()
        frames=[]
        for pointer in sorted(self.root.glob("????/??/current.json")):
            m=json.loads(pointer.read_text());path=pointer.parent/m["file"]
            if sha256_file(path)!=m["sha256"]:raise ValueError("archive checksum mismatch")
            frames.append(pd.read_parquet(path))
        return merge_archive(pd.DataFrame(),pd.concat(frames,ignore_index=True)) if frames else pd.DataFrame()

    def ingest(self,incoming,*,as_of,source_identifier="MT5_COPY_RATES_RANGE"):
        cutoff=pd.Timestamp(as_of)
        if cutoff.tzinfo is None:raise ValueError("aware cutoff required")
        if incoming.empty:return dict(status="NO_DATA",new_rows=0)
        ts=pd.DatetimeIndex(incoming.timestamp_utc)
        if ts.tz is None:raise ValueError("aware source timestamps required")
        incoming=incoming[ts+pd.Timedelta("1min")<=cutoff].copy()
        if incoming.empty:return dict(status="NO_COMPLETED_DATA",new_rows=0)
        # Identical overlap is deduplicated before strict validation; conflict is fatal.
        existing=self.read();merged=merge_archive(existing,incoming)
        from .storage import validation_view
        validate_bars(validation_view(merged),"M1")
        self.root.mkdir(parents=True,exist_ok=True)
        lock=self.root/".writer.lock"
        handle=open(lock,"a+b")
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b"0");handle.flush()
        handle.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            handle.close();raise ValueError("archive writer already active")
        try:
            # Re-read under lock so a concurrent completed writer cannot be lost.
            existing=self.read();merged=merge_archive(existing,incoming)
            for month,group in merged.groupby(merged.timestamp_utc.dt.strftime("%Y/%m"),sort=True):
                group=group.reset_index(drop=True);fingerprint=frame_hash(group)
                folder=self.root/month;folder.mkdir(parents=True,exist_ok=True)
                file=folder/(fingerprint+".parquet")
                if not file.exists():
                    temp=folder/(fingerprint+".tmp");group.to_parquet(temp,index=False);os.replace(temp,file)
                if frame_hash(pd.read_parquet(file))!=fingerprint:raise ValueError("archive revision content mismatch")
                meta=ProviderMetadata(self.provider,self.symbol,self.provider_symbol,"M1","BID","MT5_ARCHIVE")
                gaps,_=inventory(group,meta)
                previous=json.loads((folder/"current.json").read_text()) if (folder/"current.json").exists() else None
                if previous and previous["logical_fingerprint"]==fingerprint:continue
                manifest=dict(file=file.name,sha256=sha256_file(file),logical_fingerprint=fingerprint,
                              provider=self.provider,symbol=self.symbol,timeframe="M1",row_count=len(group),
                              provider_symbol=self.provider_symbol,price_type="BID",timezone="UTC",kind="RAW",
                              source_identifier=source_identifier,parent_revision_sha256=previous["sha256"] if previous else None,
                              imported_at=datetime.now(timezone.utc).isoformat(),
                              first_timestamp=group.timestamp_utc.iloc[0].isoformat(),
                              last_timestamp=group.timestamp_utc.iloc[-1].isoformat(),
                              gaps=json.loads(gaps.to_json(orient="records")))
                revision=file.with_suffix(".manifest.json")
                if not revision.exists():revision.write_text(json.dumps(manifest,indent=2))
                temp=folder/"current.tmp";temp.write_text(json.dumps(manifest,indent=2));os.replace(temp,folder/"current.json")
            return dict(status="ARCHIVED",new_rows=len(merged)-len(existing),total_rows=len(merged),
                        fingerprint=frame_hash(merged))
        finally:handle.close()  # OS releases the lock on normal exit or process failure.

    def poll(self,adapter,*,now=None,initial_days=7,overlap_minutes=60):
        now=pd.Timestamp(now or datetime.now(timezone.utc))
        if now.tzinfo is None or initial_days<1 or overlap_minutes<1:raise ValueError("invalid archive polling configuration")
        existing=self.read()
        start=(existing.timestamp_utc.max()-pd.Timedelta(f"{int(overlap_minutes)}min") if len(existing)
               else now-pd.Timedelta(f"{int(initial_days)}D"))
        incoming=adapter.rates(self.symbol,start.to_pydatetime(),now.to_pydatetime())
        return self.ingest(incoming,as_of=now,source_identifier=f"MT5:{self.provider_symbol}:{start.isoformat()}:{now.isoformat()}")
