from dataclasses import asdict
import hashlib
import json
import numpy as np
import pandas as pd
from sandbox.market_state.normalization import timeframe_minutes
from sandbox.research.canonical import canonical_json
from .gaps import inventory, SessionProfile, ContinuityPolicy

POLICY_NAME="MARKET_DATA_SANITATION_V1"
POLICY_VERSION=1
OPTIONAL=("tick_volume","spread","real_volume")
PRICES=("open","high","low","close")


def frame_hash(frame):
    """Ordered columns plus JSONL values; nulls canonicalized and datetimes UTC."""
    digest=hashlib.sha256((canonical_json(list(frame.columns))+"\n").encode())
    for row in frame.itertuples(index=False,name=None):
        values=[None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for v in row]
        digest.update((canonical_json(values)+"\n").encode())
    return digest.hexdigest()


def sanitize(raw, metadata, profile=SessionProfile(), continuity=ContinuityPolicy()):
    f=raw.copy().reset_index(drop=True)
    if not set(PRICES).issubset(f) or "timestamp_utc" not in f:
        raise ValueError("timestamp and OHLC columns required")
    original=f.copy()
    # Never localize naive values silently. Adapters must interpret declared timezone.
    valid_times=[]
    for value in f.timestamp_utc:
        try:
            t=pd.Timestamp(value)
            valid_times.append(t.tz_convert("UTC") if t.tzinfo is not None else pd.NaT)
        except (ValueError,TypeError):valid_times.append(pd.NaT)
    f["timestamp_utc"]=pd.to_datetime(valid_times,utc=True)
    for col in PRICES:f[col]=pd.to_numeric(f[col],errors="coerce")
    for col in OPTIONAL:
        f[col]=pd.to_numeric(f[col],errors="coerce") if col in f else np.nan
    reason=pd.Series("VALID",index=f.index)
    def mark(mask,label):
        reason.loc[mask & reason.eq("VALID")]=label
    mark(f.timestamp_utc.isna(),"INVALID_TIMESTAMP")
    mark((~np.isfinite(f[list(PRICES)])).any(axis=1),"NONFINITE_PRICE")
    mark((f[list(PRICES)]<=0).any(axis=1),"NONPOSITIVE_PRICE")
    mark((f.high<f[["open","close"]].max(axis=1))|(f.low>f[["open","close"]].min(axis=1))|(f.high<f.low),"INVALID_OHLC")
    minutes=timeframe_minutes(metadata.timeframe)
    if minutes<1:raise ValueError("positive timeframe required")
    mark(f.timestamp_utc.ne(f.timestamp_utc.dt.floor(f"{minutes}min")),"OFF_GRID")
    for col in OPTIONAL:
        present=original[col].notna() & original[col].astype(str).ne("") if col in original else pd.Series(False,index=f.index)
        mark(present & (f[col].isna()|~np.isfinite(f[col])|(f[col]<0)),"OTHER_INVALID")
    # Compare all observations at a timestamp before excluding invalid observations.
    duplicate=f.timestamp_utc.notna() & f.timestamp_utc.duplicated(keep=False)
    for _,g in f[duplicate].groupby("timestamp_utc",sort=True):
        payload=g[list(PRICES)+list(OPTIONAL)]
        if len(payload.drop_duplicates())==1:
            good=g.index[reason.loc[g.index].eq("VALID")]
            reason.loc[good[1:]]="DUPLICATE_IDENTICAL"
        else:
            reason.loc[g.index]="DUPLICATE_CONFLICT"
    keep=reason.eq("VALID")
    retained=f.loc[keep,["timestamp_utc",*PRICES,*OPTIONAL]].sort_values("timestamp_utc",kind="stable").reset_index(drop=True)
    for key in ("symbol","provider","provider_symbol","timeframe","price_type","source_dataset_id"):
        retained[key]=getattr(metadata,key)
    retained["quality_status"]="VALID"
    unreliable=f.loc[~keep & ~reason.eq("DUPLICATE_IDENTICAL"),"timestamp_utc"].tolist()
    gaps,segments=inventory(retained,metadata,profile,continuity,unreliable)
    # Unlocatable exclusions cannot establish reliable continuity: no indicator state survives.
    if reason.eq("INVALID_TIMESTAMP").any():
        segments=list(range(len(retained)))
    retained["continuity_segment_id"]=segments
    audit=pd.DataFrame(dict(source_row=np.arange(len(f))+1,timestamp_utc=f.timestamp_utc,classification=reason,
                            retained=keep))
    audit["original_observation"]=None
    for i in audit.index[~keep]:
        audit.at[i,"original_observation"]=json.dumps(
            {str(k):None if pd.isna(v) else str(v) for k,v in original.loc[i].items()},sort_keys=True)
    counts={str(k):int(v) for k,v in reason.value_counts().items()}
    manifest=dict(policy_name=POLICY_NAME,policy_version=POLICY_VERSION,
                  original_row_count=len(f),retained_row_count=len(retained),excluded_row_count=int((~keep).sum()),
                  exclusions_by_reason={k:v for k,v in counts.items() if k!="VALID"},
                  classification_counts=counts,parent_raw_dataset_id=metadata.source_dataset_id,
                  **{k:getattr(metadata,k) for k in ("provider","symbol","provider_symbol","timeframe","price_type","timezone")},
                  first_timestamp=retained.timestamp_utc.iloc[0].isoformat() if len(retained) else None,
                  last_timestamp=retained.timestamp_utc.iloc[-1].isoformat() if len(retained) else None,
                  continuous_segments=len(set(segments)),unlocatable_timestamp_policy="RESET_EVERY_RETAINED_BAR",session_profile=asdict(profile),
                  continuity_policy=asdict(continuity),sanitized_content_fingerprint=frame_hash(retained),
                  audit_fingerprint=frame_hash(audit),gap_fingerprint=frame_hash(gaps),
                  gap_counts={} if gaps.empty else {str(k):int(v) for k,v in gaps.classification.value_counts().items()},
                  validation_status="RESEARCH_SAFE_WITH_DOCUMENTED_GAPS" if len(gaps) else "CLEAN")
    return retained,audit,gaps,manifest
