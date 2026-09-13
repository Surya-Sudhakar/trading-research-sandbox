from __future__ import annotations
import json,math,sqlite3
from datetime import datetime,timezone
from pathlib import Path
from statistics import NormalDist
from uuid import uuid4
import numpy as np
import pandas as pd
from sandbox import STATISTICAL_EVIDENCE_ENGINE_VERSION
from sandbox.provenance import sha256_file
from sandbox.research.canonical import canonical_json,sha256_canonical
from sandbox.research.errors import ResearchError
from sandbox.statistics.models import ResearchMode,StatisticalEvidenceReport

DDL="""CREATE TABLE IF NOT EXISTS statistical_evidence_reports(report_id TEXT PRIMARY KEY,run_id TEXT NOT NULL,experiment_id TEXT,candidate_id TEXT,partition_id TEXT,research_mode TEXT NOT NULL,ledger_checksum TEXT NOT NULL,report_fingerprint TEXT NOT NULL UNIQUE,record_json TEXT NOT NULL,path TEXT NOT NULL,created_at TEXT NOT NULL);"""

def _clean(v):
    return None if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v
def max_drawdown(values):
    equity=np.r_[0.0,np.cumsum(np.asarray(values,dtype=float))];peaks=np.maximum.accumulate(equity);dd=peaks-equity;i=int(np.argmax(dd));start=int(np.argmax(equity[:i+1]));recovery=next((j for j in range(i+1,len(equity)) if equity[j]>=equity[start]),None)
    return {"maximum_drawdown_R":float(dd[i]),"drawdown_start_trade":start,"drawdown_trough_trade":i,"recovery_trade":recovery,"recovered":recovery is not None,"longest_drawdown_duration_trades":int(max((j-int(np.argmax(equity[:j+1])) for j in range(len(equity))),default=0))}
def wilson(wins,n,confidence=.95):
    if n<=0:return {"status":"NO_BINARY_TRADES","observed_win_rate":None,"lower":None,"upper":None,"n":0,"confidence_level":confidence,"method":"WILSON_SCORE"}
    z=NormalDist().inv_cdf(1-(1-confidence)/2);p=wins/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    lower=max(0,c-h);upper=min(1,c+h)
    if abs(lower)<1e-15:lower=0.0
    if abs(upper-1)<1e-15:upper=1.0
    return {"status":"OK","observed_win_rate":p,"lower":lower,"upper":upper,"n":n,"confidence_level":confidence,"method":"WILSON_SCORE"}
def streak(values,target):
    best=cur=0
    for x in values:
        cur=cur+1 if x==target else 0;best=max(best,cur)
    return best
def profile(group,min_group=1):
    r=group["net_r"].dropna().astype(float);wins=int((r>0).sum());losses=int((r<0).sum());gross_win=float(r[r>0].sum());gross_loss=float(-r[r<0].sum());n=len(r)
    return {"trades":n,"wins":wins,"losses":losses,"breakeven":int((r==0).sum()),"win_rate":wins/(wins+losses) if wins+losses else None,"average_R":float(r.mean()) if n else None,"total_R":float(r.sum()),"profit_factor":gross_win/gross_loss if gross_loss else "UNDEFINED_NO_LOSSES","max_drawdown_R":max_drawdown(r)["maximum_drawdown_R"],"interpretation":"REPORTABLE" if n>=min_group else "INSUFFICIENT_FOR_INTERPRETATION"}

class StatisticalEvidenceEngine:
    def __init__(self,registry=None,config_path:Path|None=None,results_root:Path|None=None):
        self.registry=registry;self.config_path=config_path or Path("config/statistical_evidence.json");self.config=json.loads(self.config_path.read_text(encoding="utf-8"));self.config_fingerprint=sha256_canonical(self.config);self.results_root=results_root or (registry.results_root if registry else Path("results"))/"evidence"
        if registry:
            with registry.catalog.connection() as con:con.executescript(DDL)
    def _resolved(self,frame):return frame[(frame.status.astype(str)=="CLOSED") & frame.net_r.notna()].copy().sort_values(["exit_timestamp","trade_id"],na_position="last")
    def _bootstrap(self,r,seed,block=False):
        iterations=int(self.config["bootstrap_iterations"]);rng=np.random.default_rng(seed);n=len(r)
        if not n:return {"status":"NO_TRADES","lower":None,"upper":None,"iterations":iterations,"seed":seed}
        if block and n<2*int(self.config["block_length"]):return {"status":"BLOCK_BOOTSTRAP_NOT_INFORMATIVE","lower":None,"upper":None,"iterations":iterations,"seed":seed}
        means=[];dds=[];ends=[];bl=min(int(self.config["block_length"]),n)
        for _ in range(iterations):
            if block:
                starts=rng.integers(0,n,size=math.ceil(n/bl));sample=np.concatenate([np.take(r,np.arange(s,s+bl)%n) for s in starts])[:n]
            else:sample=rng.choice(r,size=n,replace=True)
            means.append(float(np.mean(sample)));dds.append(max_drawdown(sample)["maximum_drawdown_R"]);ends.append(float(np.sum(sample)))
        q=lambda a,p:float(np.quantile(a,p));alpha=(1-float(self.config["confidence_level"]))/2
        return {"status":"OK","lower":q(means,alpha),"upper":q(means,1-alpha),"iterations":iterations,"seed":seed,"method":"MOVING_BLOCK_BOOTSTRAP" if block else "IID_NONPARAMETRIC_BOOTSTRAP","max_drawdown_percentiles":{"50":q(dds,.5),"90":q(dds,.9),"95":q(dds,.95),"99":q(dds,.99)},"ending_R_percentiles":{"5":q(ends,.05),"50":q(ends,.5),"95":q(ends,.95)}}
    def analyze_frame(self,frame,run_id,ledger_checksum,research_mode="DISCOVERY",experiment_id=None,candidate_id=None,partition_id=None,search_burden=None,evaluation_spec=None):
        mode=ResearchMode(research_mode);f=frame.copy();resolved=self._resolved(f);r=resolved.net_r.astype(float).to_numpy();binary=resolved[resolved.net_r!=0];wins=int((binary.net_r>0).sum());losses=int((binary.net_r<0).sum());n=len(r);warnings=[]
        labels=self.config["sample_labels"];label="VERY_SMALL" if n<labels["very_small"] else "SMALL" if n<labels["small"] else "MODERATE" if n<labels["moderate"] else "LARGE"
        if label in ("VERY_SMALL","SMALL"):warnings.append(label+"_SAMPLE")
        ts=pd.to_datetime(resolved.exit_timestamp,utc=True,errors="coerce") if n else pd.Series([],dtype="datetime64[ns, UTC]");years=sorted(ts.dropna().dt.year.unique().tolist());calendar_span=(ts.max()-ts.min()).total_seconds()/86400 if len(ts.dropna())>1 else 0
        sample={"total_trade_attempts":len(f),"valid_trades":int((f.status.astype(str)!="REJECTED").sum()),"rejected_trades":int((f.status.astype(str)=="REJECTED").sum()),"resolved_trades":n,"unresolved_trades":int((f.status.astype(str).isin(["OPEN","PENDING"])).sum()),"wins":wins,"losses":losses,"breakevens":int((resolved.net_r==0).sum()),"long_trades":int((resolved.direction.astype(str)=="LONG").sum()),"short_trades":int((resolved.direction.astype(str)=="SHORT").sum()),"first_trade_timestamp":ts.min().isoformat() if len(ts.dropna()) else None,"last_trade_timestamp":ts.max().isoformat() if len(ts.dropna()) else None,"calendar_span_days":calendar_span,"active_trading_years":len(years),"trades_per_active_year":n/len(years) if years else None,"trades_per_calendar_year":n/(calendar_span/365.2425) if calendar_span else None,"sample_label":label}
        p=profile(resolved);p.update({"loss_rate":losses/(wins+losses) if wins+losses else None,"median_R":float(np.median(r)) if n else None,"average_winning_R":float(np.mean(r[r>0])) if (r>0).any() else None,"average_losing_R":float(np.mean(r[r<0])) if (r<0).any() else None,"payoff_ratio":float(np.mean(r[r>0])/-np.mean(r[r<0])) if (r>0).any() and (r<0).any() else None,"longest_winning_streak":streak(np.sign(r),1),"longest_losing_streak":streak(np.sign(r),-1)})
        positive_values=np.unique(r[r>0]);negative_values=np.unique(r[r<0]);p["nominal_break_even_win_rate"]=float(-negative_values[0]/(positive_values[0]-negative_values[0])) if len(positive_values)==len(negative_values)==1 else None;p["nominal_break_even_status"]="FIXED_BINARY_OUTCOMES" if p["nominal_break_even_win_rate"] is not None else "NOT_APPLICABLE"
        ci=wilson(wins,wins+losses,float(self.config["confidence_level"]));
        if ci["lower"] is not None and ci["upper"]-ci["lower"]>self.config["wide_interval_threshold"]:warnings.append("WIDE_WIN_RATE_INTERVAL")
        seed=int(self.config["random_seed"]);iid=self._bootstrap(r,seed);block=self._bootstrap(r,seed,True)
        if iid.get("lower") is not None and iid["upper"]-iid["lower"]>1: warnings.append("WIDE_EXPECTANCY_INTERVAL")
        def ac(x):
            a=np.asarray(x[:-1],dtype=float);b=np.asarray(x[1:],dtype=float)
            value=float(np.corrcoef(a,b)[0,1]) if len(x)>=3 and np.std(a)>0 and np.std(b)>0 else None
            return value if value is not None and math.isfinite(value) else None
        rac=ac(r);oac=ac(np.sign(r));
        if any(x is not None and abs(x)>=self.config["dependence_threshold"] for x in (rac,oac)):warnings.append("SERIAL_DEPENDENCE")
        dep={"lag1_R_autocorrelation":rac,"lag1_outcome_autocorrelation":oac,"longest_winning_streak":p["longest_winning_streak"],"longest_losing_streak":p["longest_losing_streak"]}
        yearly=[]
        for y in years:
            g=resolved[ts.dt.year==y];row={"year":int(y),**profile(g,int(self.config["minimum_group_size_for_reporting"]))};row["year_completeness"]="FULL_YEAR" if ts.min().year<y<ts.max().year or (ts[ts.dt.year==y].min().dayofyear<=7 and ts[ts.dt.year==y].max().dayofyear>=358) else "PARTIAL_YEAR";yearly.append(row)
        if yearly and all(x["year_completeness"]=="PARTIAL_YEAR" for x in yearly):warnings.append("PARTIAL_YEAR_ONLY")
        if any(x["interpretation"]!="REPORTABLE" for x in yearly):warnings.append("INSUFFICIENT_GROUP_SIZE")
        positive=sum(max(0,x["total_R"]) for x in yearly);best_share=max((max(0,x["total_R"])/positive for x in yearly),default=0) if positive else None
        if best_share is not None and best_share>=self.config["concentration_thresholds"]["temporal"]:warnings.append("TEMPORAL_CONCENTRATION")
        counts=[x["trades"] for x in yearly];temporal={"yearly":yearly,"trade_frequency":{"mean_trades_per_year":float(np.mean(counts)) if counts else None,"median_trades_per_year":float(np.median(counts)) if counts else None,"minimum_trades_per_year":min(counts) if counts else None,"maximum_trades_per_year":max(counts) if counts else None,"minimum_full_year_trades":min((x["trades"] for x in yearly if x["year_completeness"]=="FULL_YEAR"),default=None)},"best_year_share_of_positive_R":best_share,"profitable_years":sum(x["total_R"]>0 for x in yearly),"losing_years":sum(x["total_R"]<0 for x in yearly)}
        symbols=[{"symbol":str(k),**profile(g)} for k,g in resolved.groupby("symbol",sort=True)];directions=[{"direction":str(k),**profile(g)} for k,g in resolved.groupby("direction",sort=True)]
        sym_share=max((max(0,x["total_R"])/sum(max(0,y["total_R"]) for y in symbols) for x in symbols),default=0) if any(x["total_R"]>0 for x in symbols) else None
        if len(symbols)>1 and sym_share>=self.config["concentration_thresholds"]["symbol"]:warnings.append("SYMBOL_CONCENTRATION")
        if len(directions)>1:
            dshare=max(x["trades"] for x in directions)/n if n else 0
            if dshare>=self.config["concentration_thresholds"]["direction"]:warnings.append("DIRECTION_CONCENTRATION")
        sorted_r=np.sort(r)[::-1];total=float(r.sum()) if n else 0;contrib=lambda k:float(sorted_r[:k].sum()/total) if total else None
        conc={"best_year_share_of_positive_R":best_share,"largest_symbol_share_of_positive_R":sym_share,"top_1_trade_contribution":contrib(1),"top_5_trade_contribution":contrib(min(5,n)),"top_10_percent_trade_contribution":contrib(max(1,math.ceil(n*.1))) if n else None,"total_R_after_removing_best_trade":float(total-sorted_r[0]) if n else None}
        if conc["top_1_trade_contribution"] is not None and conc["top_1_trade_contribution"]>=self.config["concentration_thresholds"]["outlier"]:warnings.append("OUTLIER_DEPENDENCE")
        overlap={"trades_overlapping_another_symbol":0,"percentage_overlapping":0.0,"maximum_concurrent_trades":0}
        if n and {"entry_timestamp","exit_timestamp"}.issubset(resolved):
            intervals=[]
            for _,x in resolved.iterrows():
                if pd.notna(x.entry_timestamp) and pd.notna(x.exit_timestamp):intervals.append((pd.Timestamp(x.entry_timestamp),pd.Timestamp(x.exit_timestamp),str(x.symbol)))
            flagged=set();mx=0
            for i,a in enumerate(intervals):
                concurrent=1
                for j,b in enumerate(intervals):
                    if i!=j and a[2]!=b[2] and a[0]<b[1] and b[0]<a[1]:flagged.add(i);concurrent+=1
                mx=max(mx,concurrent)
            overlap={"trades_overlapping_another_symbol":len(flagged),"percentage_overlapping":len(flagged)/n,"maximum_concurrent_trades":mx}
            if flagged:warnings.append("CROSS_SYMBOL_DEPENDENCE_POSSIBLE")
        burden=search_burden or {"experiments_attempted_in_family":0,"hypotheses_tested":0,"parameter_variants_inspected":0,"result_driven_descendants":0,"subgroup_analyses_performed":len(symbols)+len(directions)+len(yearly)}
        if burden.get("subgroup_analyses_performed",0)>1:warnings.append("MULTIPLE_SUBGROUP_INSPECTION")
        if burden.get("experiments_attempted_in_family",0)>=self.config["high_search_burden"]:warnings.append("HIGH_DISCOVERY_SEARCH_BURDEN")
        criteria=self.evaluate_criteria(evaluation_spec or {},p,ci,temporal)
        dd=max_drawdown(r);rng=np.random.default_rng(seed);perms=[max_drawdown(rng.permutation(r))["maximum_drawdown_R"] for _ in range(int(self.config["bootstrap_iterations"]))] if n else []
        dd.update({"order_sensitivity":{"label":"ORDER-SENSITIVITY ANALYSIS","observed_max_DD":dd["maximum_drawdown_R"],"permutation_median_max_DD":float(np.median(perms)) if perms else None,"permutation_95th_max_DD":float(np.quantile(perms,.95)) if perms else None,"seed":seed}})
        warnings=tuple(sorted(set(warnings)));unc={"analysis_design":"EXPLORATORY" if mode==ResearchMode.DISCOVERY else "PROSPECTIVE","win_rate":ci,"expectancy_iid_bootstrap":iid,"expectancy_block_bootstrap":block,"evaluation_criteria":criteria,"frequentist_note":"In repeated use, this interval procedure covers the true parameter at the stated long-run rate; it is not a posterior probability statement."}
        summary=f"Observed {n} resolved trades, WR {p['win_rate'] if p['win_rate'] is not None else 'undefined'}, expectancy {p['average_R'] if p['average_R'] is not None else 'undefined'}R. {label} {mode.value.lower()} evidence; warnings: {', '.join(warnings) or 'none'}. This is evidence analysis, not proof or a future-market prediction."
        basis={"run_id":run_id,"ledger_checksum":ledger_checksum,"mode":mode.value,"experiment_id":experiment_id,"candidate_id":candidate_id,"partition_id":partition_id,"config_fingerprint":self.config_fingerprint,"engine_version":STATISTICAL_EVIDENCE_ENGINE_VERSION,"sample":sample,"performance":p,"uncertainty":unc,"temporal":temporal,"symbols":symbols,"directions":directions,"concentration":conc,"dependence":dep,"drawdown":dd,"resampling":{"iid":iid,"block":block},"warnings":warnings,"search_burden":burden,"overlap":overlap}
        fp=sha256_canonical(basis);return StatisticalEvidenceReport(report_id="EVID-"+fp[:20],run_id=run_id,experiment_id=experiment_id,candidate_id=candidate_id,partition_id=partition_id,research_mode=mode,ledger_checksum=ledger_checksum,statistical_config_fingerprint=self.config_fingerprint,created_at=datetime.now(timezone.utc),sample_profile=sample,performance_profile=p,uncertainty_profile=unc,temporal_profile=temporal,symbol_profile={"pooled":p,"symbols":symbols,"overlap":overlap},direction_profile={"directions":directions},concentration_profile={**conc,"search_burden":burden},dependence_profile=dep,drawdown_profile=dd,resampling_profile={"label":"RESAMPLED SCENARIOS","iid":iid,"block":block},warnings=warnings,evidence_summary=summary,report_fingerprint=fp)
    def evaluate_criteria(self,spec,p,ci,temporal):
        out=[]
        for key,value in sorted(spec.items()):
            if key=="minimum_win_rate":actual=p["win_rate"]
            elif key=="minimum_wilson_lower_bound":actual=ci["lower"]
            elif key=="minimum_net_r":actual=p["total_R"]
            elif key=="minimum_trades_per_year":actual=temporal["trade_frequency"]["mean_trades_per_year"]
            else:continue
            out.append({"criterion":key,"required":value,"actual":actual,"result":"INCONCLUSIVE" if actual is None else "CRITERION_MET" if actual>=value else "CRITERION_NOT_MET"})
        return out
    def analyze_run(self,run_id,research_mode="DISCOVERY",experiment_id=None,candidate_id=None,partition_id=None,evaluation_spec=None):
        if not self.registry:raise ResearchError("registry required")
        mode=ResearchMode(research_mode)
        if mode!=ResearchMode.DISCOVERY and (not partition_id or not candidate_id):raise ResearchError("PROTECTED_EVIDENCE_ANALYSIS_DENIED: candidate and partition binding required")
        if partition_id:
            from sandbox.partition.service import PartitionService
            from sandbox.partition.models import AccessContext,AccessOperation
            context={ResearchMode.DISCOVERY:AccessContext.DISCOVERY_CONTEXT,ResearchMode.VALIDATION:AccessContext.VALIDATION_CONTEXT,ResearchMode.FINAL_TEST:AccessContext.FINAL_TEST_CONTEXT}[mode]
            PartitionService(self.registry).access(partition_id,context,AccessOperation.EVALUATION_READ,actor="statistical-evidence-engine",candidate_id=candidate_id,experiment_id=experiment_id)
        with self.registry.catalog.connection() as con:row=con.execute("SELECT * FROM research_runs WHERE run_id=?",(run_id,)).fetchone()
        if not row:raise ResearchError("run not found")
        try:
            with self.registry.catalog.connection() as con:sealed=con.execute("SELECT 1 FROM sealed_final_results WHERE json_extract(result_json,'$.run_id')=?",(run_id,)).fetchone()
            if sealed:raise ResearchError("FINAL_TEST_ACCESS_DENIED: sealed evidence may only be created by the authorized final runner")
        except sqlite3.OperationalError:pass
        row=dict(row);path=Path(row["ledger_path"]);path=path if path.is_absolute() else self.registry.project_root/path;summary_path=Path(row["summary_path"]);summary_path=summary_path if summary_path.is_absolute() else self.registry.project_root/summary_path;summary=json.loads(summary_path.read_text(encoding="utf-8"));checksum=sha256_file(path)
        if checksum!=summary["ledger_checksum"]:raise ResearchError("BLOCK EVIDENCE ANALYSIS: ledger checksum mismatch")
        if experiment_id and evaluation_spec is None:evaluation_spec=self.registry.get_experiment(experiment_id).evaluation_spec
        burden={"experiments_attempted_in_family":0,"hypotheses_tested":0,"parameter_variants_inspected":0,"result_driven_descendants":0,"subgroup_analyses_performed":0}
        if candidate_id:
            from sandbox.control.service import ResearchControl
            raw=ResearchControl(self.registry).burden(candidate_id);burden.update({"experiments_attempted_in_family":raw["number_of_experiments"],"hypotheses_tested":raw["number_of_hypotheses_considered"],"parameter_variants_inspected":raw["number_of_parameter_variants"],"result_driven_descendants":raw["number_of_result_driven_modifications"]})
        self.registry.append_journal("EVIDENCE_ANALYSIS_STARTED",self.registry.get_experiment(experiment_id).program_id if experiment_id else "SYSTEM",f"Evidence analysis started for {run_id}",experiment_id=experiment_id,run_id=run_id)
        report=self.analyze_frame(pd.read_parquet(path),run_id,checksum,mode.value,experiment_id,candidate_id,partition_id,burden,evaluation_spec);self.store(report);return report
    def store(self,report):
        self.results_root.mkdir(parents=True,exist_ok=True);path=self.results_root/f"{report.report_id}.json";serialized=json.dumps(report.model_dump(mode="json"),indent=2,sort_keys=True);path.write_text(serialized,encoding="utf-8")
        if self.registry:
            with self.registry.catalog.connection() as con:con.execute("INSERT OR IGNORE INTO statistical_evidence_reports VALUES (?,?,?,?,?,?,?,?,?,?,?)",(report.report_id,report.run_id,report.experiment_id,report.candidate_id,report.partition_id,report.research_mode.value,report.ledger_checksum,report.report_fingerprint,report.model_dump_json(),str(path),report.created_at.isoformat()))
            program=self.registry.get_experiment(report.experiment_id).program_id if report.experiment_id else "SYSTEM";self.registry.append_journal("EVIDENCE_REPORT_CREATED",program,f"Evidence report {report.report_id} created",experiment_id=report.experiment_id,run_id=report.run_id,metadata={"report_fingerprint":report.report_fingerprint,"ledger_checksum":report.ledger_checksum})
        return path
    def get(self,report_id):
        if not self.registry:raise ResearchError("registry required")
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM statistical_evidence_reports WHERE report_id=?",(report_id,)).fetchone()
        return StatisticalEvidenceReport.model_validate_json(row[0]) if row else None
    def sensitivity(self,variants,parameter_name):
        values=sorted(variants,key=lambda x:x["parameter_value"]);tested=[x["parameter_value"] for x in values];expect=[x["expectancy"] for x in values]
        if len(values)<3:status="INSUFFICIENT_EVIDENCE";cliff=False
        else:
            diffs=[abs(expect[i]-expect[i-1]) for i in range(1,len(expect))];scale=max(abs(float(np.median(expect))),.1);cliff=max(diffs)>scale*1.5;status="CLIFF_LIKE" if cliff else "BROADLY_STABLE" if all(x>=0 for x in expect) and max(diffs)<=scale*.5 else "VARIABLE"
        return {"parameter_name":parameter_name,"tested_values":tested,"registered_variants":values,"status":status,"warnings":["PARAMETER_CLIFF"] if cliff else [],"generated_parameter_values":[]}
    def cost_sensitivity(self,registered_scenarios):
        rows=sorted(registered_scenarios,key=lambda x:x["adversity_rank"]);base=next((x for x in rows if x["name"]=="BASELINE"),None)
        if not base:raise ResearchError("registered BASELINE scenario required")
        fragile=any(x["expectancy"]<=0 for x in rows if x["adversity_rank"]>base["adversity_rank"]);return {"registered_scenarios":rows,"baseline":"BASELINE","status":"COST_FRAGILE" if fragile else "SURVIVES_REGISTERED_STRESS","warnings":["COST_FRAGILITY"] if fragile else []}
