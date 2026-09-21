"""First frozen-model temporal validation; no fitting, tuning or model selection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_data.sanitation import frame_hash
from sandbox.partition.models import AccessContext, AccessOperation, PartitionRole
from sandbox.provenance import sha256_file
from .frozen_model import FrozenMarketStateModel
from .gmm import GaussianMixtureStateModel
from .scaler import DiscoveryStandardScaler
from .future_behavior import FUTURE_HORIZONS, STATES, METRICS, STEP, _guard, _statistics, _effect, measure_window
from .interpretation import DEFAULT_FEATURE_PATH
from .validation_contract import validate_contract, CONTRACT_SHA256
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_FEATURE_SCHEMA_VERSION

EXPECTED_MODEL_SHA256 = "6895652563cf698c88d5f6c8d4145ec4502350a5a54f79f58d3e034648eb0b55"
START = datetime(2023,1,1,tzinfo=timezone.utc)
END = datetime(2025,1,1,tzinfo=timezone.utc)
DEFAULT_MODEL = Path("results/market_state/frozen_market_state_model_v1/model.json")
DEFAULT_CONTRACT = Path("results/market_state/future_behavior_validation_contract_v1.json")
DEFAULT_OUTPUT = Path("results/market_state/future_behavior_validation_v1.json")


@contextmanager
def no_fitting():
    attempts = {"fit_calls":0}
    def forbidden(*args,**kwargs):
        attempts["fit_calls"] += 1
        raise RuntimeError("fitting is prohibited during validation")
    with ExitStack() as stack:
        for cls in (GaussianMixture,StandardScaler,GaussianMixtureStateModel,DiscoveryStandardScaler):
            for name in ("fit","fit_predict","fit_transform","partial_fit"):
                if hasattr(cls,name):
                    stack.enter_context(patch.object(cls,name,forbidden))
        yield attempts


def load_frozen_inputs(contract_path=DEFAULT_CONTRACT,model_path=DEFAULT_MODEL):
    raw = Path(contract_path).read_bytes()
    contract = json.loads(raw)
    validate_contract(contract)
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen model whole-file SHA256 mismatch")
    # The externally pinned whole-file digest authenticates these provenance bindings.
    metadata = json.loads(Path(model_path).read_text(encoding="utf-8"))["state"]["provenance"]
    model = FrozenMarketStateModel.load(model_path,artifact_sha256=EXPECTED_MODEL_SHA256,
        feature_artifact_sha256=contract["feature_artifact"]["sha256"],
        discovery_input_sha256=metadata["discovery_input_sha256"],
        feature_names=contract["feature_schema"]["ordered_names"],
        schema_version=contract["feature_schema"]["version"])
    if metadata["discovery_artifact_sha256"] != contract["discovery_artifact"]["sha256"]:
        raise ValueError("model/discovery provenance mismatch")
    return contract, model, hashlib.sha256(raw).hexdigest()


def _window(frame, timestamp, *, price=False):
    _guard(frame.columns)
    if frame.columns.duplicated().any() or timestamp not in frame:
        raise ValueError("unique columns and timestamp required")
    series = frame[timestamp]
    if not isinstance(series.dtype,pd.DatetimeTZDtype):
        raise ValueError("timezone-aware timestamp column required")
    if series.isna().any():
        raise ValueError("missing timestamps")
    times = series.dt.tz_convert("UTC")
    # Price rows are open-timestamped; their closes must also be inside validation.
    ceiling = pd.Timestamp(END)-STEP if price else pd.Timestamp(END)
    selected = frame.loc[(times>=START)&(times<ceiling)].copy()
    selected[timestamp] = times.loc[selected.index]
    selected = selected.sort_values(timestamp,kind="stable").reset_index(drop=True)
    if selected[timestamp].duplicated().any() or not selected[timestamp].eq(selected[timestamp].dt.floor("15min")).all():
        raise ValueError("unique M15-aligned timestamps required")
    if "continuity_segment_id" not in selected or selected.continuity_segment_id.isna().any():
        raise ValueError("explicit continuity segment IDs required")
    for name, expected in (("provider","DUKASCOPY"),("symbol","EURUSD"),("decision_timeframe","M15")):
        if name in selected and not selected[name].eq(expected).all():
            raise ValueError(f"unexpected {name}; same Dukascopy EURUSD M15 lineage required")
    return selected


def _feature_rows(features):
    order = tuple(c for c in features.columns if c in MARKET_STATE_FEATURE_NAMES_V1)
    if order != MARKET_STATE_FEATURE_NAMES_V1:
        raise ValueError("exact frozen feature order required")
    selected = _window(features,"timestamp")
    total = len(selected)
    selected = selected.dropna(subset=list(MARKET_STATE_FEATURE_NAMES_V1)).reset_index(drop=True)
    if selected.empty:
        raise ValueError("no eligible validation feature rows")
    if not np.isfinite(selected.loc[:,MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite frozen features")
    return selected,total-len(selected)


def _confidence(values):
    return {"mean":float(np.mean(values)) if len(values) else None,
            "median":float(np.median(values)) if len(values) else None,
            "p10":float(np.quantile(values,.1)) if len(values) else None,
            "p90":float(np.quantile(values,.9)) if len(values) else None}


def _occupancy(labels):
    return {state:{"count":int(np.sum(labels==i)),"fraction":float(np.mean(labels==i)) if len(labels) else None,
                   "percent":float(100*np.mean(labels==i)) if len(labels) else None} for i,state in enumerate(STATES)}


def evaluate_hypotheses(summaries,comparisons):
    result = {}
    for hypothesis,metric in (("H1_MAGNITUDE","absolute_return"),("H2_VOLATILITY","realized_volatility")):
        per_horizon = {}
        for h in FUTURE_HORIZONS:
            means = {s:summaries[s][str(h)]["continuous"][metric]["mean"] for s in STATES}
            enough = all(v is not None for v in means.values())
            yes = bool(enough and means["S0"]>means["S1"] and means["S0"]>means["S2"])
            per_horizon[str(h)] = {"minutes":h*15,"means":means,
                "comparisons":{pair:comparisons[str(h)][pair][metric] for pair in ("S0_minus_S1","S0_minus_S2")},
                "criterion_satisfied":"YES" if yes else "NO", "classification":"replicated direction" if yes else "failed direction",
                "insufficient_evidence":not enough}
        count = sum(x["criterion_satisfied"]=="YES" for x in per_horizon.values())
        result[hypothesis] = {"horizons":per_horizon,"replicated_horizons":count,"total_horizons":5,"replication":f"{count} / 5"}
    h3 = {}
    for h in FUTURE_HORIZONS:
        h3[str(h)] = {}
        for a,b in combinations(STATES,2):
            pair = f"{a}_minus_{b}"
            effects = {name:comparisons[str(h)][pair][name]["cohens_d"] for name in ("signed_return","absolute_return","realized_volatility")}
            signed = effects["signed_return"]
            ratios = {}
            for name in ("absolute_return","realized_volatility"):
                denominator = effects[name]
                valid = signed is not None and denominator is not None and denominator!=0
                ratios[name] = {"absolute_effect_ratio":abs(signed)/abs(denominator) if valid else None,
                    "signed_effect_smaller":abs(signed)<abs(denominator) if signed is not None and denominator is not None else None,
                    "undefined_reason":None if valid else "undefined effect or zero comparator effect"}
            h3[str(h)][pair] = {"cohens_d":effects,"descriptive_ratios":ratios}
    result["H3_DIRECTION"] = {"assessment":"DESCRIPTIVE_ONLY_NO_BINARY_MATERIALITY_VERDICT","horizons":h3}
    return result


def analyze_validation(features,prices,model,contract):
    """Pure synthetic-testable analysis; real entry point supplies verified frozen artifacts."""
    validate_contract(contract)
    metadata = model.metadata
    if metadata["feature_schema_version"] != MARKET_STATE_FEATURE_SCHEMA_VERSION or tuple(metadata["feature_names"]) != MARKET_STATE_FEATURE_NAMES_V1:
        raise ValueError("model feature schema/order mismatch")
    if metadata["state_ids"] != list(STATES):
        raise ValueError("model state order mismatch")
    for key,value in contract["model_configuration"].items():
        if key!="class" and metadata["gmm"]["configuration"][key]!=value:
            raise ValueError("model configuration mismatch")
    ready,missing = _feature_rows(features)
    prices = _window(prices,"timestamp_utc",price=True)
    required = {"open","high","low","close"}
    if not required.issubset(prices.columns):
        raise ValueError("OHLC columns required")
    bars = tuple(OHLCBar(r.timestamp_utc.to_pydatetime(),r.open,r.high,r.low,r.close) for r in prices.itertuples())
    x = ready.loc[:,MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)
    with no_fitting() as guard:
        labels = model.predict(x,feature_names=MARKET_STATE_FEATURE_NAMES_V1)
        proba = model.predict_proba(x,feature_names=MARKET_STATE_FEATURE_NAMES_V1)
    if (labels.shape!=(len(ready),) or not np.isin(labels,[0,1,2]).all() or proba.shape!=(len(ready),3)
            or not np.isfinite(proba).all() or (proba<0).any() or (proba>1).any()
            or not np.allclose(proba.sum(axis=1),1,rtol=0,atol=1e-12)
            or not np.array_equal(labels,proba.argmax(axis=1))):
        raise ValueError("invalid frozen model predictions")
    confidence = proba.max(axis=1)
    connected = (ready.timestamp.diff().eq(STEP)&ready.continuity_segment_id.eq(ready.continuity_segment_id.shift())).to_numpy()
    transitions = np.zeros((3,3),dtype=np.int64)
    np.add.at(transitions,(labels[:-1][connected[1:]],labels[1:][connected[1:]]),1)
    outgoing = transitions.sum(axis=1)
    transition_probabilities = np.divide(transitions,outgoing[:,None],out=np.zeros((3,3)),where=outgoing[:,None]!=0)
    buckets = {(s,h):{metric:[] for metric in METRICS} for s in STATES for h in FUTURE_HORIZONS}
    reasons = ("missing_reference","gap_or_segment_boundary","insufficient_future_bars","validation_end_boundary")
    exclusions = {key:{reason:0 for reason in reasons} for key in buckets}
    directions = {key:{direction:0 for direction in ("UP","DOWN","FLAT")} for key in buckets}
    positions = {bar.open_time_utc+STEP:i for i,bar in enumerate(bars)}
    segments = tuple(prices.continuity_segment_id)
    for row,(stamp,segment,label) in enumerate(zip(ready.timestamp,ready.continuity_segment_id,labels)):
        state = STATES[int(label)]
        position = positions.get(stamp)
        if position is None:
            for h in FUTURE_HORIZONS:exclusions[state,h]["missing_reference"]+=1
            continue
        if segments[position]!=segment:
            raise ValueError("feature/price continuity identity mismatch")
        available = []
        reason = "insufficient_future_bars"
        for offset in range(1,max(FUTURE_HORIZONS)+1):
            j = position+offset
            if j>=len(bars):break
            if bars[j].open_time_utc!=bars[position].open_time_utc+offset*STEP or segments[j]!=segment:
                reason="gap_or_segment_boundary"
                break
            available.append(bars[j])
        valid = tuple(h for h in FUTURE_HORIZONS if stamp+h*STEP<END and h<=len(available))
        measured = measure_window(bars[position],available,valid) if valid else {}
        for h in FUTURE_HORIZONS:
            if stamp+h*STEP>=END:
                exclusions[state,h]["validation_end_boundary"]+=1
            elif h not in measured:
                exclusions[state,h][reason]+=1
            else:
                record = measured[h]
                for metric in METRICS:buckets[state,h][metric].append(record[metric])
                direction = "UP" if record["signed_return"]>0 else "DOWN" if record["signed_return"]<0 else "FLAT"
                directions[state,h][direction]+=1
    summaries = {s:{} for s in STATES}
    for (s,h),columns in buckets.items():
        count = len(columns["signed_return"])
        summaries[s][str(h)] = {"observation_count":count,"excluded_counts":exclusions[s,h],
            "contract_excluded_counts":{"missing_reference":exclusions[s,h]["missing_reference"],
                "gap_or_segment_boundary":exclusions[s,h]["gap_or_segment_boundary"],
                "end_of_validation_data":exclusions[s,h]["insufficient_future_bars"]+exclusions[s,h]["validation_end_boundary"]},
            "continuous":{name:_statistics(values) for name,values in columns.items()},
            "direction":{name:{"count":n,"fraction":n/count if count else None} for name,n in directions[s,h].items()}}
    comparisons = {str(h):{f"{a}_minus_{b}":{name:_effect(buckets[a,h][name],buckets[b,h][name]) for name in METRICS[:5]}
                  for a,b in combinations(STATES,2)} for h in FUTURE_HORIZONS}
    return {
        "experiment":"MARKET_STATE_FUTURE_BEHAVIOUR_VALIDATION_V1",
        "feature_schema_version":MARKET_STATE_FEATURE_SCHEMA_VERSION,"ordered_feature_names":list(MARKET_STATE_FEATURE_NAMES_V1),
        "validation_start_inclusive":START.isoformat(),"validation_end_exclusive":END.isoformat(),
        "model_configuration":contract["model_configuration"],"state_ids":list(STATES),
        "no_refit":True,"fit_call_count":guard["fit_calls"],
        "eligible_validation_observations":len(ready),"assigned_observations":len(labels),"incomplete_feature_rows_excluded":missing,
        "state_occupancy":_occupancy(labels),
        "yearly_occupancy":{str(year):_occupancy(labels[ready.timestamp.dt.year.to_numpy()==year]) for year in (2023,2024)},
        "posterior_confidence":{"overall":_confidence(confidence),"per_state":{s:_confidence(confidence[labels==i]) for i,s in enumerate(STATES)}},
        "transitions":{"counts":transitions.tolist(),"probabilities":transition_probabilities.tolist(),"outgoing_counts":outgoing.tolist(),"state_order":list(STATES)},
        "states":summaries,"comparisons":comparisons,"hypotheses":evaluate_hypotheses(summaries,comparisons),
        "horizon_counts":{str(h):{"minutes":h*15,"observations":sum(len(buckets[s,h]["signed_return"]) for s in STATES),
                        "exclusions":{reason:sum(exclusions[s,h][reason] for s in STATES) for reason in reasons}} for h in FUTURE_HORIZONS},
        "continuity_rule":contract["continuity"],"formulas":contract["formulas"],
        "input_provenance":{"eligible_features_sha256":frame_hash(ready),"validation_prices_sha256":frame_hash(prices),
                            "assigned_labels_sha256":hashlib.sha256(np.asarray(labels,dtype="<i8").tobytes()).hexdigest()},
        "frozen_model_limitation":"The earlier Discovery model's per-row assignment digest was unavailable. The canonical frozen model was reconstructed under the frozen specification and reproduced S0=4755, S1=44972, S2=43935. Exact canonical serialization/load equivalence was verified; equality to unavailable original per-row assignments cannot be proved.",
        "scientific_limits":["Overlapping future windows are dependent; no independence-based significance claim.","H3 is descriptive only; no materiality threshold or profitable-state selection."],
        "final_holdout_observations_used":0,
    }


def read_validation_inputs(service,feature_path,contract):
    """Provenance metadata/checksums first, predicate-filtered observations second."""
    feature_path = Path(feature_path)
    metadata_path = feature_path.with_name("metadata.json")
    if sha256_file(Path(contract["discovery_artifact"]["path"])) != contract["discovery_artifact"]["sha256"]:
        raise ValueError("discovery artifact checksum mismatch")
    binding = contract["feature_artifact"]
    if sha256_file(feature_path)!=binding["sha256"] or sha256_file(metadata_path)!=binding["metadata_sha256"]:
        raise ValueError("feature artifact/metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["provider"]!="DUKASCOPY" or metadata["source_timeframe"]!="M15" or metadata["symbol"]!="EURUSD":
        raise ValueError("same Dukascopy EURUSD M15 source required")
    lineage = [x for x in metadata["transformation_history"] if x["step"]=="AUTHORIZED_DISCOVERY"]
    if len(lineage)!=1:raise ValueError("unique original source lineage required")
    manifest = service.get_partition(lineage[0]["partition_id"])
    if (manifest is None or manifest.role not in (PartitionRole.DISCOVERY,PartitionRole.VALIDATION)
            or manifest.source_dataset_id!=metadata["source_dataset_id"] or manifest.timeframe!="M15" or manifest.symbol!="EURUSD"
            or manifest.partition_fingerprint!=metadata["source_partition_fingerprint"]
            or manifest.source_dataset_checksum!=metadata["source_dataset_checksum"]):
        raise ValueError("source partition provenance mismatch or protected role")
    schema = pq.read_schema(feature_path)
    _guard(schema.names)
    columns = ["timestamp",*MARKET_STATE_FEATURE_NAMES_V1,"continuity_segment_id","provider","source_dataset_id","symbol","decision_timeframe"]
    if not set(columns).issubset(schema.names):raise ValueError("required frozen artifact schema missing")
    context = AccessContext.DISCOVERY_CONTEXT if manifest.role==PartitionRole.DISCOVERY else AccessContext.VALIDATION_CONTEXT
    prices = service.access(manifest.partition_id,context,AccessOperation.FEATURE_ANALYSIS,actor="market-state-validation-v1",
                            start_timestamp=START,end_timestamp=END-STEP)
    # PartitionService authenticates bounded reads internally without exposing protected paths.
    features = pd.read_parquet(feature_path,columns=columns,filters=[("timestamp",">=",START),("timestamp","<",END)])
    if not features.source_dataset_id.eq(manifest.source_dataset_id).all():raise ValueError("feature row lineage mismatch")
    return features,prices,{"provider":metadata["provider"],"source_dataset_id":manifest.source_dataset_id,
        "partition_id":manifest.partition_id,"storage_partition_role":manifest.role.value,"scientific_role":"VALIDATION",
        "partition_fingerprint":manifest.partition_fingerprint,"partition_sha256":manifest.partition_checksum,
        "feature_artifact_sha256":binding["sha256"],"feature_metadata_sha256":binding["metadata_sha256"],
        "observation_reads":"Parquet predicates applied at read: validation features only; price opens >=2023 and closes <2025. No feature recomputation.",
        "integrity_reads":"Byte-level whole-file SHA256 verification only; no out-of-window observation statistics."}


def _publish_json(path,payload):
    # Atomic create-only publication, never overwriting a prior result or attempt.
    import os,tempfile
    encoded = (json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);stream.write(encoded)
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def run_validation(service,feature_path=DEFAULT_FEATURE_PATH,contract_path=DEFAULT_CONTRACT,model_path=DEFAULT_MODEL,output=DEFAULT_OUTPUT):
    output=Path(output)
    if output.exists():raise FileExistsError("validation result exists; no repeat evaluation permitted")
    with no_fitting() as guard:
        contract,model,contract_file_sha=load_frozen_inputs(contract_path,model_path)
        attempt=output.with_suffix(".attempt.json")
        _publish_json(attempt,{"experiment":"MARKET_STATE_FUTURE_BEHAVIOUR_VALIDATION_V1","model_sha256":EXPECTED_MODEL_SHA256,
                               "contract_sha256":CONTRACT_SHA256,"status":"ATTEMPT_RESERVED_NOT_COMPLETION",
                               "serialization_format":"MARKET_STATE_PARAMETERS_V1", "model_metadata":model.metadata})
        features,prices,source=read_validation_inputs(service,feature_path,contract)
        result=analyze_validation(features,prices,model,contract)
        if sha256_file(model_path)!=EXPECTED_MODEL_SHA256 or sha256_file(contract_path)!=contract_file_sha:
            raise ValueError("frozen inputs changed during validation")
        if (sha256_file(Path(feature_path))!=contract["feature_artifact"]["sha256"]
                or sha256_file(Path(feature_path).with_name("metadata.json"))!=contract["feature_artifact"]["metadata_sha256"]):
            raise ValueError("feature artifact changed during validation")
        result.update(frozen_model_sha256=EXPECTED_MODEL_SHA256,validation_contract_sha256=CONTRACT_SHA256,
                      validation_contract_file_sha256=contract_file_sha,source_provenance=source,fit_call_count=guard["fit_calls"],
                      implementation_sha256=sha256_file(Path(__file__)))
        result["result_content_sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        checksum=_publish_json(output,result)
    print(json.dumps({"output":str(output),"whole_file_sha256":checksum,"status":"COMPLETE"}),flush=True)
    return result,checksum


def main(argv=None):
    from sandbox.partition.service import PartitionService
    from sandbox.research.registry import ResearchRegistry
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,default=Path("data/catalog.sqlite3"))
    args=parser.parse_args(argv)
    run_validation(PartitionService(ResearchRegistry(args.catalog)))


if __name__=="__main__":main()
