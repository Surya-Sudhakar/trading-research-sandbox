"""H4 exploratory transition discovery using immutable GMM assignments, never fit."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

from sandbox.market_data.sanitation import frame_hash
from sandbox.provenance import sha256_file
from . import validation as frozen
from .future_behavior import _effect
from .validation_contract import validate_contract, CONTRACT_SHA256
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1 as FEATURES
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_SCHEMA_VERSION

VERSION = "MARKET_STATE_TRANSITION_DISCOVERY_V1"
HORIZONS = (1, 2, 4, 8, 12)
STATES = ("S0", "S1", "S2")
COMPARE = ("max_posterior", "posterior_margin", "posterior_entropy", "state_age_bars",
           "bars_since_transition", "delta_max_posterior", "delta_posterior_entropy")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
OUTPUT = Path("results/market_state/state_transition_discovery_v1.json")
PROTECTED = (frozen.DEFAULT_MODEL, frozen.DEFAULT_CONTRACT,
             Path("results/market_state/future_behavior_v1.json"),
             Path("results/market_state/future_behavior_validation_v1.json"),
             Path("results/market_state/future_behavior_validation_v1.attempt.json"),
             Path("results/market_state/state_interpretation_v1.json"),
             Path("results/market_state/gmm_v1.json"))


def specification():
    """Declared before outcomes are opened; no result-dependent configuration."""
    return {
        "version": VERSION, "hypothesis": "H4", "status": "EXPLORATORY_DISCOVERY_NOT_INDEPENDENT_REPLICATION",
        "start_inclusive": frozen.START.isoformat(), "end_exclusive": frozen.END.isoformat(),
        "horizons_bars": list(HORIZONS), "horizons_minutes": [15*h for h in HORIZONS],
        "state_ids": list(STATES), "feature_order": list(FEATURES),
        "state_age_bars": "One-based observed current-state run length; resets on change or disconnection. Initial runs are left-censored, not known true ages.",
        "previous_state": "State at immediately preceding contiguous eligible observation; null at a disconnection.",
        "bars_since_transition": "Zero at an observed change, then increments; null until a change is observed in this continuous block.",
        "deltas": "Current minus immediately preceding contiguous eligible observation; null at a disconnection.",
        "entropy": "-sum(p*ln(p)), with 0*ln(0)=0; natural logarithms.",
        "outcome": "Any different state in future steps 1..h; full continuous horizon required even if a change is already observed.",
        "destination": "First different state within the complete horizon; descriptive frequencies among TRANSITION only.",
        "continuity": "Exact 15-minute spacing AND unchanged recorded segment ID; missing feature vectors break history and outcomes.",
        "comparison": "TRANSITION minus STAY; sample standard deviations; pooled-sample-SD Cohen's d, null if undefined.",
        "predictors_compared": list(COMPARE), "missing_predictors": "Per-predictor complete cases; no imputation; report missing counts by outcome.",
        "bootstrap": {"method": "whole continuous eligible-observation cluster bootstrap",
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence": .95,
            "interval": "2.5/97.5 percentile intervals for mean difference and Cohen's d",
            "resampling": "Resample all continuous blocks with replacement, preserving every within-block row and outcome. Shared weights across comparisons.",
            "limitations": "Assumes approximately independent blocks; does not remove dependence across blocks or multiplicity. Null intervals if fewer than two contributing blocks or two defined replicates."},
        "classifier": False, "threshold_optimization": False, "pass_fail_threshold": None,
    }


def causal_predictors(timestamps, segments, probabilities, labels):
    """Past-only history construction. Never accepts future outcome columns."""
    times = pd.DatetimeIndex(timestamps)
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    segments = np.asarray(segments)
    n = len(times)
    if not n or times.tz is None or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("ordered unique aware observation timestamps required")
    times = times.tz_convert("UTC")
    if ((times < frozen.START).any() or (times >= frozen.END).any()
            or not times.equals(times.floor("15min"))):
        raise ValueError("observations must be aligned M15 within 2023-2024")
    if segments.shape != (n,) or pd.isna(segments).any():
        raise ValueError("explicit continuity IDs required")
    if (p.shape != (n,3) or labels.shape != (n,) or not np.isfinite(p).all()
            or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-12)
            or not np.isin(labels,[0,1,2]).all() or not np.array_equal(labels,p.argmax(axis=1))):
        raise ValueError("invalid frozen posterior/state outputs")
    connected = np.zeros(n,dtype=bool)
    connected[1:] = (np.diff(times.asi8)==pd.Timedelta(frozen.STEP).value)&(segments[1:]==segments[:-1])
    blocks = np.cumsum(~connected)-1
    sorted_p = np.sort(p,axis=1)
    maxima = sorted_p[:,-1]
    logarithms = np.zeros_like(p)
    np.log(p,out=logarithms,where=p>0)
    entropy = -np.sum(p*logarithms,axis=1)
    age = np.ones(n,dtype=int)
    since = np.full(n,np.nan)
    previous = np.full(n,None,dtype=object)
    for i in range(1,n):
        if not connected[i]: continue
        previous[i] = STATES[int(labels[i-1])]
        if labels[i] != labels[i-1]:
            since[i] = 0
        else:
            age[i] = age[i-1]+1
            since[i] = since[i-1]+1
    def delta(values):
        result = np.full(n,np.nan)
        result[1:] = np.diff(values)
        result[~connected] = np.nan
        return result
    result = pd.DataFrame({"timestamp":times,"state_t":[STATES[int(x)] for x in labels],
        "p_s0":p[:,0],"p_s1":p[:,1],"p_s2":p[:,2],"max_posterior":maxima,
        "posterior_margin":sorted_p[:,-1]-sorted_p[:,-2],"posterior_entropy":entropy,
        "state_age_bars":age,"previous_state":previous,"bars_since_transition":since,
        "delta_p_s0":delta(p[:,0]),"delta_p_s1":delta(p[:,1]),"delta_p_s2":delta(p[:,2]),
        "delta_max_posterior":delta(maxima),"delta_posterior_entropy":delta(entropy)})
    return result, blocks


def transition_outcomes(predictors, blocks):
    """Separate future-label table; no mutation or addition of targets to predictors."""
    labels = predictors.state_t.to_numpy()
    times = pd.DatetimeIndex(predictors.timestamp)
    blocks = np.asarray(blocks)
    n = len(labels)
    if blocks.shape != (n,): raise ValueError("block alignment mismatch")
    result = {}
    for h in HORIZONS:
        outcome = np.full(n,np.nan)
        destination = np.full(n,None,dtype=object)
        excluded = np.full(n,None,dtype=object)
        for i in range(n):
            if times[i]+h*frozen.STEP >= frozen.END:
                excluded[i]="period_end_boundary"
            elif i+h>=n:
                # If another block occurs before the truncated tail, this is a gap.
                excluded[i]="gap_or_segment_boundary" if blocks[-1]!=blocks[i] else "insufficient_future_observations"
            elif blocks[i+h]!=blocks[i]:
                excluded[i]="gap_or_segment_boundary"
            else:
                changes=np.flatnonzero(labels[i+1:i+h+1]!=labels[i])
                outcome[i]=int(len(changes)>0)
                if len(changes): destination[i]=labels[i+1+changes[0]]
        result[str(h)]=pd.DataFrame({"transition_within_h":outcome,"first_destination":destination,"exclusion":excluded})
    return result


def _stats(x):
    x = np.asarray(x,dtype=float)
    return {"n":len(x),"mean":float(x.mean()) if len(x) else None,
        "median":float(np.median(x)) if len(x) else None,
        "standard_deviation":float(x.std(ddof=1)) if len(x)>1 else None}


def bootstrap_intervals(values, outcomes, blocks, mask, weights):
    """Cluster sufficient statistics reproduce full-row resampling without row copies."""
    groups = weights.shape[1]
    moments=[]
    for label in (0,1):
        use=mask & (outcomes==label) & np.isfinite(values)
        ids=blocks[use]; x=values[use]
        moments.append([np.bincount(ids,weights=w,minlength=groups) for w in (np.ones(len(x)),x,x*x)])
    contributing=sum((moments[0][0]+moments[1][0])>0)
    if contributing<2:
        return {"difference_in_means":None,"cohens_d":None,"defined_difference_replicates":0,"defined_effect_replicates":0,"contributing_blocks":int(contributing)}
    n0,s0,q0=[weights@m for m in moments[0]]
    n1,s1,q1=[weights@m for m in moments[1]]
    with np.errstate(divide="ignore",invalid="ignore"):
        difference=s1/n1-s0/n0
        pooled=((q1-s1*s1/n1)+(q0-s0*s0/n0))/(n1+n0-2)
        effect=difference/np.sqrt(np.maximum(pooled,0))
    difference[(n0==0)|(n1==0)]=np.nan
    effect[(n0<2)|(n1<2)|(pooled<=0)]=np.nan
    def interval(values):
        valid=values[np.isfinite(values)]
        return ([float(x) for x in np.quantile(valid,[.025,.975])] if len(valid)>=2 else None),len(valid)
    ci_diff,count_diff=interval(difference);ci_d,count_d=interval(effect)
    return {"difference_in_means":ci_diff,"cohens_d":ci_d,"defined_difference_replicates":count_diff,
        "defined_effect_replicates":count_d,"contributing_blocks":int(contributing)}


def summarize(predictors, blocks, outcomes):
    groups=int(blocks.max())+1
    weights=np.random.default_rng(BOOTSTRAP_SEED).multinomial(groups,np.full(groups,1/groups),size=BOOTSTRAP_REPLICATES).astype(float)
    result={}
    for horizon,table in outcomes.items():
        y=table.transition_within_h.to_numpy()
        per_state={}
        for state in ("ALL",*STATES):
            scope=np.ones(len(y),dtype=bool) if state=="ALL" else predictors.state_t.eq(state).to_numpy()
            valid=scope & np.isfinite(y)
            n_stay=int(np.sum(valid & (y==0)));n_transition=int(np.sum(valid & (y==1)))
            comparisons={}
            for name in COMPARE:
                x=predictors[name].to_numpy(dtype=float)
                stay=x[valid & (y==0) & np.isfinite(x)];transition=x[valid & (y==1) & np.isfinite(x)]
                effect=_effect(transition.tolist(),stay.tolist())
                comparisons[name]={"stay":_stats(stay),"transition":_stats(transition),
                    "missing_stay":n_stay-len(stay),"missing_transition":n_transition-len(transition),
                    **effect,"bootstrap_95_ci":bootstrap_intervals(x,y,blocks,valid,weights)}
            per_state[state]={"n_stay":n_stay,"n_transition":n_transition,
                "transition_probability":n_transition/(n_stay+n_transition) if n_stay+n_transition else None,
                "first_destination":{s:{"count":int(np.sum(valid & table.first_destination.eq(s).to_numpy())),
                    "fraction_of_transitions":float(np.sum(valid & table.first_destination.eq(s).to_numpy())/n_transition) if n_transition else None} for s in STATES},
                "exclusions":{reason:int(np.sum(scope & table.exclusion.eq(reason).to_numpy())) for reason in
                    ("gap_or_segment_boundary","insufficient_future_observations","period_end_boundary")},
                "comparisons":comparisons}
        result[horizon]={"minutes":15*int(horizon),"groups":per_state}
    return result


def analyze(features, model, contract):
    validate_contract(contract)
    metadata=model.metadata
    if (metadata["feature_schema_version"]!=MARKET_STATE_FEATURE_SCHEMA_VERSION
            or tuple(metadata["feature_names"])!=FEATURES or metadata["state_ids"]!=list(STATES)):
        raise ValueError("frozen feature/state identity mismatch")
    if any(metadata["gmm"]["configuration"][k]!=value for k,value in contract["model_configuration"].items() if k!="class"):
        raise ValueError("frozen model configuration mismatch")
    ready,missing=frozen._feature_rows(features)
    with frozen.no_fitting() as guard:
        x=ready.loc[:,FEATURES].to_numpy(dtype=float)
        labels=model.predict(x,feature_names=FEATURES)
        probabilities=model.predict_proba(x,feature_names=FEATURES)
    if guard["fit_calls"]: raise RuntimeError("fit attempt detected during H4")
    predictors,blocks=causal_predictors(ready.timestamp,ready.continuity_segment_id,probabilities,labels)
    outcomes=transition_outcomes(predictors,blocks)
    return {"experiment":VERSION,"specification":specification(),
        "eligible_observations":len(ready),"incomplete_feature_rows_excluded":missing,
        "continuous_observation_blocks":int(blocks.max())+1,"disconnections_between_eligible_observations":int(blocks.max()),
        "initial_left_censored_run_observations":int(predictors.bars_since_transition.isna().sum()),
        "state_occupancy":{s:int(predictors.state_t.eq(s).sum()) for s in STATES},
        "predictor_fields":list(predictors.columns),"horizons":summarize(predictors,blocks,outcomes),
        "provenance":{"eligible_features_sha256":frame_hash(ready),"causal_predictors_sha256":frame_hash(predictors),
            "outcomes_sha256":{h:frame_hash(table) for h,table in outcomes.items()}},
        "fit_call_counts":{"total":guard["fit_calls"],"scaler":0,"gmm":0},
        "holdout_observations_used":0,
        "limitations":[
            "H4 is exploratory on the previously evaluated 2023-2024 period, not independent confirmation of H4.",
            "Initial state runs are left-censored; state age is observed run length. Time since transition is unknown until an observed transition.",
            "After an observed transition, bars_since_transition equals state_age_bars minus one; these are not independent predictors.",
            "Overlapping horizons are dependent. Whole-block bootstrap assumes approximate independence between blocks; intervals are pointwise, without multiplicity correction.",
            "Posterior confidence and assigned state share the same GMM; associations are descriptive and do not establish classifier performance or causality.",
            "Only complete feature vectors and complete future state horizons are included; no imputation or gap bridging.",
            "Original Discovery per-row assignment digest was unavailable. The canonical reconstruction reproduced Discovery counts; serialization equivalence was exact, original per-row equivalence cannot be proved."]}


def frozen_snapshot():
    paths=set(PROTECTED)|set(frozen.DEFAULT_MODEL.parent.glob("*.json"))
    return {str(p):sha256_file(p) for p in sorted(paths)}


def report_markdown(result):
    lines=["# Market State Transition Discovery V1", "", "Exploratory H4; all differences are TRANSITION minus STAY. No pass/fail threshold.", "",
           "Mean, median and standard deviation are shown as (mean; median; SD). Full counts, missingness, first destinations and bootstrap diagnostics are in the JSON.", ""]
    def number(value): return "null" if value is None else f"{value:.6g}"
    def stats(value): return "; ".join(number(value[k]) for k in ("mean","median","standard_deviation"))
    def interval(value): return "null" if value is None else "["+", ".join(number(x) for x in value)+"]"
    for state in ("ALL",*STATES):
        lines += [f"## {state}", "", "| Minutes | Predictor | N stay / transition | STAY mean; median; SD | TRANSITION mean; median; SD | Mean difference [95% CI] | Cohen d [95% CI] |",
                  "|---:|---|---:|---|---|---|---|"]
        for h in HORIZONS:
            group=result["horizons"][str(h)]["groups"][state]
            for name,c in group["comparisons"].items():
                lines.append(f"| {h*15} | {name} | {c['stay']['n']} / {c['transition']['n']} | {stats(c['stay'])} | {stats(c['transition'])} | {number(c['difference_in_means'])} {interval(c['bootstrap_95_ci']['difference_in_means'])} | {number(c['cohens_d'])} {interval(c['bootstrap_95_ci']['cohens_d'])} |")
        lines.append("")
    lines += ["## Limitations", "", *["- "+x for x in result["limitations"]], ""]
    return "\n".join(lines)


def run(service, output=OUTPUT):
    output=Path(output)
    attempt=output.with_suffix(".attempt.json")
    report=output.with_suffix(".md")
    if output.exists() or attempt.exists() or report.exists():
        raise FileExistsError("H4 attempt/result exists; refusing repeat execution")
    with frozen.no_fitting() as guard:
        contract,model,contract_file_sha=frozen.load_frozen_inputs()
        before=frozen_snapshot()
        frozen._publish_json(attempt,{"experiment":VERSION,"status":"ATTEMPT_RESERVED_NOT_COMPLETION",
            "specification":specification(),"frozen_files_before":before,"contract_checksum":CONTRACT_SHA256,
            "model_metadata":model.metadata,"implementation_sha256":sha256_file(Path(__file__))})
        features,prices,source=frozen.read_validation_inputs(service,frozen.DEFAULT_FEATURE_PATH,contract)
        # The existing reader authenticates source lineage and continuity. No OHLC outcomes are calculated.
        ready,_=frozen._feature_rows(features)
        reference=prices.set_index(prices.timestamp_utc+frozen.STEP).continuity_segment_id
        expected=reference.reindex(pd.DatetimeIndex(ready.timestamp))
        if expected.isna().any() or not np.array_equal(expected.to_numpy(),ready.continuity_segment_id.to_numpy()):
            raise ValueError("feature/source continuity identity mismatch")
        result=analyze(features,model,contract)
        after=frozen_snapshot()
        if before!=after: raise ValueError("protected artifact changed during H4 execution")
        for path,expected_hash in ((frozen.DEFAULT_FEATURE_PATH,contract["feature_artifact"]["sha256"]),
                (frozen.DEFAULT_FEATURE_PATH.with_name("metadata.json"),contract["feature_artifact"]["metadata_sha256"])):
            if sha256_file(path)!=expected_hash: raise ValueError("feature artifact changed during H4 execution")
        validate_contract(json.loads(frozen.DEFAULT_CONTRACT.read_text()))
        result.update(frozen_files_before=before,frozen_files_after=after,contract_checksum_before=CONTRACT_SHA256,
            contract_checksum_after=CONTRACT_SHA256,contract_file_sha256=contract_file_sha,source_provenance=source,
            fit_call_counts={"total":guard["fit_calls"],"scaler":0,"gmm":0},implementation_sha256=sha256_file(Path(__file__)))
        if guard["fit_calls"]: raise RuntimeError("fit attempt detected during H4")
        checksum=frozen._publish_json(output,result)
        # Derived readable tables; exclusively the published result, no additional analysis.
        with report.open("x",encoding="utf-8") as stream: stream.write(report_markdown(result))
        if sha256_file(output)!=checksum: raise ValueError("published result checksum mismatch")
    print(json.dumps({"status":"COMPLETE","output":str(output),"sha256":checksum}),flush=True)
    return result,checksum


def main(argv=None):
    from sandbox.partition.service import PartitionService
    from sandbox.research.registry import ResearchRegistry
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,default=Path("data/catalog.sqlite3"))
    args=parser.parse_args(argv)
    run(PartitionService(ResearchRegistry(args.catalog)))


if __name__=="__main__": main()
