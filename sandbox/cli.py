from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sandbox.catalog import Catalog
from sandbox.config import Settings
from sandbox.history import download, inspect_history
from sandbox.integrity import audit as audit_market_data
from sandbox.logging import configure
from sandbox.mt5 import MT5Adapter, MT5Error
from sandbox.storage import RawStore
from sandbox.execution.artifacts import store_run
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import ExecutionConfig, TradeIntent
from sandbox.research.models import ExperimentStatus, HypothesisStatus, VerdictValue
from sandbox.research.registry import ResearchRegistry
from sandbox.research.errors import ResearchError
from sandbox.audit.auditor import ResearchAuditor
from sandbox.audit.models import AuditPhase
from sandbox.audit.report import write_audit_report
from sandbox.control.service import ResearchControl
from sandbox.control.models import StrategyFamilyStatus
from sandbox.partition.service import PartitionService
from sandbox.partition.models import AccessContext,AccessOperation,PartitionRole
from sandbox.statistics.service import StatisticalEvidenceEngine
from sandbox.external_import import ExternalImportGateway,ImportSpec,Resolution,PriceType,TimestampFormat,ImportError as ExternalImportError
from sandbox.strategies.registry import default_registry


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sandbox", description="Read-only IC Markets MT5 data foundation")
    groups = root.add_subparsers(dest="group", required=True)
    mt5 = groups.add_parser("mt5").add_subparsers(dest="command", required=True)
    mt5.add_parser("status")
    symbols = groups.add_parser("symbols").add_subparsers(dest="command", required=True)
    search = symbols.add_parser("search"); search.add_argument("pair")
    history = groups.add_parser("history").add_subparsers(dest="command", required=True)
    inspect = history.add_parser("inspect"); inspect.add_argument("pair")
    get = history.add_parser("download"); get.add_argument("pair"); get.add_argument("--timeframe", default="M1", choices=["M1"]); get.add_argument("--start", type=_date); get.add_argument("--end", type=_date); get.add_argument("--chunk-days", type=int)
    data = groups.add_parser("data").add_subparsers(dest="command", required=True)
    scan = data.add_parser("audit"); scan.add_argument("target", help="Parquet path or broker symbol")
    data.add_parser("catalog")
    import_file=data.add_parser("import-file");import_file.add_argument("--file",type=Path,action="append",required=True);import_file.add_argument("--provider",required=True);import_file.add_argument("--symbol",required=True);import_file.add_argument("--canonical-symbol");import_file.add_argument("--resolution",choices=[x.value for x in Resolution],required=True);import_file.add_argument("--price-type",choices=[x.value for x in PriceType],required=True);import_file.add_argument("--timezone",required=True);import_file.add_argument("--target-execution-broker");import_file.add_argument("--derive-mid",action="store_true");import_file.add_argument("--interpretation-reason");import_file.add_argument("--timestamp-column",default="timestamp");import_file.add_argument("--open-column",default="open");import_file.add_argument("--high-column",default="high");import_file.add_argument("--low-column",default="low");import_file.add_argument("--close-column",default="close");import_file.add_argument("--bid-column",default="bid");import_file.add_argument("--ask-column",default="ask");import_file.add_argument("--tick-volume-column");import_file.add_argument("--real-volume-column");import_file.add_argument("--spread-column");import_file.add_argument("--bid-volume-column");import_file.add_argument("--ask-volume-column")
    import_file.add_argument("--timestamp-format",choices=[x.value for x in TimestampFormat],default=TimestampFormat.ISO8601.value)
    for command in ("import-status","import-inspect"):
        action=data.add_parser(command);action.add_argument("import_id")
    data.add_parser("import-list")
    certify=data.add_parser("certify");certify.add_argument("dataset_id")
    partition=data.add_parser("partition").add_subparsers(dest="partition_command",required=True)
    create_p=partition.add_parser("create");create_p.add_argument("source_dataset_id");create_p.add_argument("program_id");create_p.add_argument("role");create_p.add_argument("symbol");create_p.add_argument("timeframe");create_p.add_argument("start");create_p.add_argument("end");create_p.add_argument("--diagnostic-overlap",action="store_true")
    for command in ("inspect","verify","access-history"):
        action=partition.add_parser(command);action.add_argument("partition_id")
    backtest = groups.add_parser("backtest").add_subparsers(dest="command", required=True)
    run = backtest.add_parser("run")
    run.add_argument("dataset", type=Path); run.add_argument("intents", type=Path)
    run.add_argument("--config", type=Path); run.add_argument("--results-dir", type=Path, default=Path("results"))
    for command in ("inspect", "ledger", "summary"):
        item = backtest.add_parser(command); item.add_argument("run_id")
    research = groups.add_parser("research").add_subparsers(dest="research_group", required=True)
    for entity in ("program", "question", "hypothesis"):
        sub = research.add_parser(entity).add_subparsers(dest="research_command", required=True)
        create = sub.add_parser("create"); create.add_argument("spec", type=Path)
        if entity == "hypothesis":
            approve_h = sub.add_parser("approve"); approve_h.add_argument("hypothesis_id")
        if entity == "question":
            inspect_q=sub.add_parser("inspect");inspect_q.add_argument("question_id")
            close_q=sub.add_parser("close");close_q.add_argument("question_id");close_q.add_argument("--reason",required=True);close_q.add_argument("--by",required=True)
            reopen_q=sub.add_parser("reopen");reopen_q.add_argument("question_id");reopen_q.add_argument("--reason",required=True);reopen_q.add_argument("--new-evidence",required=True);reopen_q.add_argument("--by",required=True)
    experiment = research.add_parser("experiment").add_subparsers(dest="research_command", required=True)
    create_e=experiment.add_parser("create"); create_e.add_argument("spec",type=Path)
    for command in ("approve","start","complete","inspect","ancestry","manifest"):
        action=experiment.add_parser(command); action.add_argument("experiment_id")
    replicate=experiment.add_parser("replicate"); replicate.add_argument("experiment_id"); replicate.add_argument("run_id"); replicate.add_argument("run_fingerprint"); replicate.add_argument("--original-run-id")
    link=experiment.add_parser("link-run"); link.add_argument("experiment_id"); link.add_argument("run_id")
    verdict=research.add_parser("verdict").add_subparsers(dest="research_command",required=True)
    record_v=verdict.add_parser("record"); record_v.add_argument("experiment_id"); record_v.add_argument("result_id"); record_v.add_argument("verdict",choices=[x.value for x in VerdictValue]); record_v.add_argument("--rationale",required=True); record_v.add_argument("--decided-by",required=True)
    timeline=research.add_parser("timeline"); timeline.add_argument("program_id"); timeline.set_defaults(research_command="timeline")
    journal=research.add_parser("journal").add_subparsers(dest="research_command",required=True); journal.add_parser("verify")
    audit_cli=research.add_parser("audit").add_subparsers(dest="research_command",required=True)
    for command in ("pre","post","history"):
        action=audit_cli.add_parser(command);action.add_argument("experiment_id")
    inspect_a=audit_cli.add_parser("inspect");inspect_a.add_argument("audit_id")
    override_a=audit_cli.add_parser("override");override_a.add_argument("audit_id");override_a.add_argument("--reason",required=True);override_a.add_argument("--approved-by",required=True)
    family=research.add_parser("family").add_subparsers(dest="research_command",required=True)
    for command in ("inspect","parameters"):
        action=family.add_parser(command);action.add_argument("family_id")
    create_f=family.add_parser("create");create_f.add_argument("spec",type=Path)
    exhaust_f=family.add_parser("exhaust");exhaust_f.add_argument("family_id");exhaust_f.add_argument("--reason",required=True)
    for name in ("status","tree","next"):
        action=research.add_parser(name);action.add_argument("program_id");action.set_defaults(research_command=name)
    branch=research.add_parser("branch").add_subparsers(dest="research_command",required=True)
    inspect_b=branch.add_parser("inspect");inspect_b.add_argument("question_id")
    close_b=branch.add_parser("close");close_b.add_argument("question_id");close_b.add_argument("--reason",required=True);close_b.add_argument("--by",required=True)
    decision=research.add_parser("decision").add_subparsers(dest="research_command",required=True)
    create_d=decision.add_parser("create");create_d.add_argument("spec",type=Path)
    inspect_d=decision.add_parser("inspect");inspect_d.add_argument("decision_id")
    record_d=decision.add_parser("record");record_d.add_argument("decision_id");record_d.add_argument("outcome")
    candidate=research.add_parser("candidate").add_subparsers(dest="research_command",required=True)
    create_c=candidate.add_parser("create");create_c.add_argument("spec",type=Path)
    for name in ("inspect","lineage","burden","card"):
        action=candidate.add_parser(name);action.add_argument("candidate_id")
    freeze_c=candidate.add_parser("freeze");freeze_c.add_argument("candidate_id");freeze_c.add_argument("--validation-dataset",action="append",required=True)
    validate_c=candidate.add_parser("validate");validate_c.add_argument("candidate_id")
    contamination=research.add_parser("contamination").add_subparsers(dest="research_command",required=True);inspect_cont=contamination.add_parser("inspect");inspect_cont.add_argument("candidate_id")
    validation=research.add_parser("validation").add_subparsers(dest="research_command",required=True)
    bind_v=validation.add_parser("bind");bind_v.add_argument("candidate_id");bind_v.add_argument("partition_id")
    run_v=validation.add_parser("run");run_v.add_argument("candidate_id");run_v.add_argument("partition_id");run_v.add_argument("intents",type=Path)
    final=research.add_parser("final").add_subparsers(dest="research_command",required=True)
    auth_f=final.add_parser("authorize");auth_f.add_argument("candidate_id");auth_f.add_argument("partition_id");auth_f.add_argument("validation_result_id");auth_f.add_argument("--authorized-by",required=True)
    run_f=final.add_parser("run");run_f.add_argument("candidate_id");run_f.add_argument("partition_id");run_f.add_argument("intents",type=Path)
    discovery=research.add_parser("discovery").add_subparsers(dest="research_command",required=True)
    inspect_d=discovery.add_parser("inspect-prepared");inspect_d.add_argument("prepared_id");inspect_d.add_argument("--root",type=Path,default=Path("results/discovery_runner"))
    execute_d=discovery.add_parser("execute");execute_d.add_argument("prepared_id");execute_d.add_argument("proposal",type=Path);execute_d.add_argument("--root",type=Path,default=Path("results/discovery_runner"));execute_d.add_argument("--pip-size",type=float,default=0.0001);execute_d.add_argument("--config",type=Path)
    stats=groups.add_parser("stats").add_subparsers(dest="command",required=True)
    analyze_s=stats.add_parser("analyze");analyze_s.add_argument("run_id");analyze_s.add_argument("--mode",choices=["DISCOVERY","VALIDATION","FINAL_TEST"],default="DISCOVERY");analyze_s.add_argument("--experiment-id");analyze_s.add_argument("--candidate-id");analyze_s.add_argument("--partition-id")
    for command in ("inspect","yearly","symbols","directions","uncertainty","drawdown"):
        action=stats.add_parser(command);action.add_argument("report_id")
    sensitivity_s=stats.add_parser("sensitivity");sensitivity_s.add_argument("variants",type=Path);sensitivity_s.add_argument("parameter")
    costs_s=stats.add_parser("costs");costs_s.add_argument("scenarios",type=Path)
    strategies=groups.add_parser("strategies").add_subparsers(dest="command",required=True);strategies.add_parser("list")
    for command in ("inspect","validate","fingerprint"):
        action=strategies.add_parser(command);action.add_argument("strategy_id");action.add_argument("--version")
    return root


def _print(value) -> None:
    print(json.dumps(value, indent=2, default=str, sort_keys=True))


def _connect(settings: Settings) -> MT5Adapter:
    adapter = MT5Adapter(settings.terminal_path)
    adapter.connect()
    return adapter

def _assert_unprotected_market_path(catalog:Catalog,path:Path)->None:
    try:
        resolved=path.resolve()
        with catalog.connection() as con:
            rows=con.execute("SELECT p.role,p.path,d.path FROM data_partitions p LEFT JOIN datasets d ON p.source_dataset_id=d.dataset_id").fetchall()
        protected=[]
        for row in rows:
            role=PartitionRole(row[0])
            if role not in {PartitionRole.VALIDATION,PartitionRole.FINAL_TEST}:continue
            for value in row[1:]:
                if not isinstance(value,str) or not value.strip() or "\x00" in value:
                    raise ValueError("invalid protection metadata")
                protected.append(Path(value).resolve())
    except Exception:
        raise ResearchError("PROTECTED_DATA_PATH_CHECK_FAILED: access denied; protection metadata unavailable") from None
    if resolved in protected:raise ResearchError("PROTECTED_DATA_PATH: use the Stage 6 controlled execution boundary")


def main(argv: list[str] | None = None) -> int:
    settings = Settings.load()
    configure(settings.log_level)
    args = parser().parse_args(argv)
    catalog = Catalog(settings.catalog_path)
    try:
        if args.group=="strategies":
            registry=default_registry()
            if args.command=="list":value=[x.model_dump(mode="json") for x in registry.list()]
            else:
                plugin=registry.resolve(args.strategy_id,args.version)
                if args.command=="inspect":value=plugin.metadata.model_dump(mode="json")
                elif args.command=="validate":value={"valid":registry.validate(plugin)}
                else:
                    params=plugin.metadata.default_parameters;fingerprint,code_fp,param_fp=registry.fingerprint(plugin,params);value={"strategy_fingerprint":fingerprint,"strategy_code_fingerprint":code_fp,"parameter_fingerprint":param_fp}
            _print(value);return 0
        if args.group=="stats":
            registry=ResearchRegistry(settings.catalog_path,Path.cwd(),Path("results"));engine=StatisticalEvidenceEngine(registry)
            if args.command=="analyze":
                value=engine.analyze_run(args.run_id,args.mode,args.experiment_id,args.candidate_id,args.partition_id).model_dump(mode="json")
            elif args.command=="sensitivity":value=engine.sensitivity(json.loads(args.variants.read_text(encoding="utf-8")),args.parameter)
            elif args.command=="costs":value=engine.cost_sensitivity(json.loads(args.scenarios.read_text(encoding="utf-8")))
            else:
                report=engine.get(args.report_id)
                if not report:raise ResearchError("evidence report not found")
                value=report.model_dump(mode="json") if args.command=="inspect" else {"yearly":report.temporal_profile["yearly"]} if args.command=="yearly" else report.symbol_profile if args.command=="symbols" else report.direction_profile if args.command=="directions" else report.uncertainty_profile if args.command=="uncertainty" else report.drawdown_profile
            _print(value);return 0
        if args.group=="data" and args.command=="partition":
            registry=ResearchRegistry(settings.catalog_path,Path.cwd(),Path("results"));service=PartitionService(registry)
            if args.partition_command=="create":value=service.create_partition(args.source_dataset_id,args.program_id,args.role,args.symbol,args.timeframe,args.start,args.end,args.diagnostic_overlap)
            elif args.partition_command=="inspect":
                value=service.access(args.partition_id,AccessContext.SYSTEM_INTEGRITY_CONTEXT,AccessOperation.METADATA_READ,actor="cli-admin")
            elif args.partition_command=="verify":value=service.access(args.partition_id,AccessContext.SYSTEM_INTEGRITY_CONTEXT,AccessOperation.CHECKSUM_VERIFY,actor="cli-integrity")
            else:value=service.access_history(args.partition_id)
            _print(value.model_dump(mode="json") if hasattr(value,"model_dump") else value);return 0
        if args.group == "research":
            registry=ResearchRegistry(settings.catalog_path,Path.cwd(),Path("results"))
            if args.research_group == "program" and args.research_command == "create":
                spec=json.loads(args.spec.read_text(encoding="utf-8")); _print(registry.create_program(**spec).model_dump(mode="json")); return 0
            if args.research_group == "question" and args.research_command == "create":
                spec=json.loads(args.spec.read_text(encoding="utf-8")); record=ResearchControl(registry).create_question(**spec) if spec.get("parent_question_id") else registry.create_question(**spec);_print(record.model_dump(mode="json")); return 0
            if args.research_group == "question":
                control=ResearchControl(registry)
                if args.research_command=="inspect":
                    record=registry.get_question(args.question_id)
                    if not record:raise ResearchError("question not found")
                    _print(record.model_dump(mode="json"))
                elif args.research_command=="close":_print(control.set_question_status(args.question_id,__import__("sandbox.research.models",fromlist=["QuestionStatus"]).QuestionStatus.CLOSED,args.reason,changed_by=args.by).model_dump(mode="json"))
                else:_print(control.reopen_question(args.question_id,args.reason,args.new_evidence,args.by).model_dump(mode="json"))
                return 0
            if args.research_group == "hypothesis":
                if args.research_command == "create":
                    spec=json.loads(args.spec.read_text(encoding="utf-8")); _print(registry.create_hypothesis(**spec).model_dump(mode="json"))
                else: _print(registry.set_hypothesis_status(args.hypothesis_id,HypothesisStatus.APPROVED).model_dump(mode="json"))
                return 0
            if args.research_group == "experiment":
                if args.research_command == "create":
                    spec=json.loads(args.spec.read_text(encoding="utf-8")); _print(registry.create_experiment(**spec).model_dump(mode="json"))
                elif args.research_command == "approve": _print(registry.approve_experiment(args.experiment_id).model_dump(mode="json"))
                elif args.research_command == "start": _print(registry.transition_experiment(args.experiment_id,ExperimentStatus.RUNNING).model_dump(mode="json"))
                elif args.research_command == "complete": _print(registry.transition_experiment(args.experiment_id,ExperimentStatus.COMPLETED).model_dump(mode="json"))
                elif args.research_command == "inspect": _print(registry.audit_context(args.experiment_id))
                elif args.research_command == "ancestry": _print(registry.experiment_ancestry(args.experiment_id))
                elif args.research_command == "replicate": _print(registry.create_replication(args.experiment_id,args.run_id,args.run_fingerprint,args.original_run_id))
                elif args.research_command == "link-run": _print(registry.link_stage2_run(args.experiment_id,args.run_id).model_dump(mode="json"))
                else:
                    paths=registry.generate_manifest(args.experiment_id); _print({"manifest_json":str(paths[0]),"manifest_markdown":str(paths[1])})
                return 0
            if args.research_group == "verdict":
                _print(registry.record_verdict(args.experiment_id,args.result_id,VerdictValue(args.verdict),args.rationale,args.decided_by).model_dump(mode="json")); return 0
            if args.research_group == "timeline": print(registry.timeline(args.program_id)); return 0
            if args.research_group == "journal": _print({"valid":registry.verify_journal_integrity()}); return 0
            if args.research_group == "audit":
                auditor=ResearchAuditor(registry)
                if args.research_command in ("pre","post"):
                    phase=AuditPhase.PRE_RUN if args.research_command=="pre" else AuditPhase.POST_RUN;audit=auditor.audit(args.experiment_id,phase);paths=write_audit_report(audit,Path("results"));_print({"audit":audit.model_dump(mode="json"),"json_report":str(paths[0]),"markdown_report":str(paths[1])})
                elif args.research_command=="inspect":
                    audit=auditor.get_audit(args.audit_id)
                    if not audit:raise ResearchError("audit not found")
                    _print(audit.model_dump(mode="json"))
                elif args.research_command=="history":_print([x.model_dump(mode="json") for x in auditor.audit_history(args.experiment_id)])
                else:_print(auditor.create_override(args.audit_id,args.reason,args.approved_by).model_dump(mode="json"))
                return 0
            if args.research_group == "family":
                auditor=ResearchAuditor(registry);control=ResearchControl(registry)
                if args.research_command=="create":spec=json.loads(args.spec.read_text(encoding="utf-8"));_print(control.create_family(**spec).model_dump(mode="json"))
                elif args.research_command=="exhaust":_print(control.set_family_status(args.family_id,StrategyFamilyStatus.EXHAUSTED,args.reason).model_dump(mode="json"))
                elif args.research_command=="parameters":_print(auditor.parameter_history(args.family_id))
                else:
                    strategy_family=control.get_family(args.family_id);_print(strategy_family.model_dump(mode="json") if strategy_family else [x.model_dump(mode="json") for x in auditor._family(args.family_id)])
                return 0
            if args.research_group in {"status","tree","next"}:
                control=ResearchControl(registry);value=control.status_report(args.program_id) if args.research_group=="status" else control.question_tree(args.program_id) if args.research_group=="tree" else control.what_next(args.program_id);_print(value);return 0
            if args.research_group=="branch":
                control=ResearchControl(registry);_print(control.branch(args.question_id) if args.research_command=="inspect" else control.close_branch(args.question_id,args.reason,args.by));return 0
            if args.research_group=="decision":
                control=ResearchControl(registry)
                if args.research_command=="create":spec=json.loads(args.spec.read_text(encoding="utf-8"));value=control.create_decision(**spec).model_dump(mode="json")
                elif args.research_command=="inspect":
                    record=control.get_decision(args.decision_id)
                    if not record:raise ResearchError("decision not found")
                    value=record.model_dump(mode="json")
                else:value=control.record_decision(args.decision_id,args.outcome).model_dump(mode="json")
                _print(value);return 0
            if args.research_group=="discovery":
                from sandbox.research.discovery_runner import (
                    load_prepared_discovery_fast, execute_discovery_proposal
                )
                service=PartitionService(registry)
                print("[0/6] Loading prepared Discovery snapshot (fast execution path)...", flush=True)
                prepared=load_prepared_discovery_fast(args.root,args.prepared_id)
                if args.research_command=="inspect-prepared":
                    packet=prepared.packet
                    value={
                        "prepared_id":prepared.prepared_id,
                        "packet_fingerprint":packet.fingerprint,
                        "target_field_id":packet.target_field_id,
                        "allowed_condition_feature_ids":list(packet.allowed_condition_feature_ids),
                        "permitted_entry_kinds":list(packet.permitted_entry_kinds),
                        "permitted_stop_kinds":list(packet.permitted_stop_kinds),
                        "permitted_target_kinds":list(packet.permitted_target_kinds),
                        "population_records":len(prepared.population.records),
                        "partition_id":prepared.population.partition_id,
                    }
                else:
                    proposal=json.loads(args.proposal.read_text(encoding="utf-8"))
                    config_payload=json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
                    if "candle_interval_seconds" not in config_payload:
                        manifest=service.get_partition(prepared.population.partition_id)
                        interval={"M1":60,"M15":900}.get(manifest.timeframe)
                        if interval is None:
                            raise ResearchError(f"UNSUPPORTED_DISCOVERY_TIMEFRAME: {manifest.timeframe}")
                        config_payload["candle_interval_seconds"]=interval
                    execution_config=ExecutionConfig.model_validate(config_payload)
                    value=execute_discovery_proposal(
                        service,prepared,prepared.packet,proposal,
                        root=args.root,pip_size=args.pip_size,execution_config=execution_config,
                        fast_path=True, progress=lambda message: print(message, flush=True),
                    )
                _print(value);return 0
            if args.research_group=="candidate":
                control=ResearchControl(registry)
                if args.research_command=="create":spec=json.loads(args.spec.read_text(encoding="utf-8"));value=control.create_candidate(**spec).model_dump(mode="json")
                elif args.research_command=="inspect":
                    record=control.get_candidate(args.candidate_id)
                    if not record:raise ResearchError("candidate not found")
                    value=record.model_dump(mode="json")
                elif args.research_command=="freeze":value=control.freeze_candidate(args.candidate_id,args.validation_dataset).model_dump(mode="json")
                elif args.research_command=="lineage":value=control.candidate_lineage(args.candidate_id)
                elif args.research_command=="burden":value=control.burden(args.candidate_id)
                elif args.research_command=="card":value=control.candidate_card(args.candidate_id)
                else:value=control.request_validation(args.candidate_id).model_dump(mode="json")
                _print(value);return 0
            if args.research_group=="contamination":_print([x.model_dump(mode="json") for x in PartitionService(registry).contamination(args.candidate_id)]);return 0
            if args.research_group=="validation":
                service=PartitionService(registry)
                if args.research_command=="bind":value=service.bind_validation(args.candidate_id,args.partition_id).model_dump(mode="json")
                else:
                    intents=[TradeIntent.model_validate(x) for x in json.loads(args.intents.read_text(encoding="utf-8"))];sealed=service.sealed_validation_execute(args.candidate_id,args.partition_id,intents,results_root=Path("results")/"validation");value={"candidate":sealed["candidate"].model_dump(mode="json"),"result":sealed["result"]}
                _print(value);return 0
            if args.research_group=="final":
                service=PartitionService(registry)
                if args.research_command=="authorize":value=service.authorize_final(args.candidate_id,args.partition_id,args.validation_result_id,args.authorized_by).model_dump(mode="json")
                else:
                    intents=[TradeIntent.model_validate(x) for x in json.loads(args.intents.read_text(encoding="utf-8"))];value=service.sealed_final_execute(args.candidate_id,args.partition_id,intents,results_root=Path("results")/"final")
                _print(value);return 0
        if args.group == "backtest":
            if args.command == "run":
                if not catalog.path.exists():PartitionService(ResearchRegistry(settings.catalog_path))
                _assert_unprotected_market_path(catalog,args.dataset)
                market = pd.read_parquet(args.dataset)
                intent_payload = json.loads(args.intents.read_text(encoding="utf-8"))
                config_payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
                intents = [TradeIntent.model_validate(item) for item in intent_payload]
                config = ExecutionConfig.model_validate(config_payload)
                result = BacktestEngine(config).run(intents, market)
                stored = store_run(result, args.results_dir)
                dataset_ids = sorted({item.dataset_id for item in intents})
                catalog.add_research_run(stored.run_id, stored.run_fingerprint, ",".join(dataset_ids), result.execution_engine_version, result.ledger_schema_version, result.metrics_version, config.model_dump(mode="json"), str(stored.ledger_path), str(stored.summary_path))
                _print({**stored.__dict__, "ledger_path": str(stored.ledger_path), "summary_path": str(stored.summary_path)})
                return 0
            record = catalog.research_run(args.run_id)
            if record is None:
                raise ValueError(f"run not found: {args.run_id}")
            if args.command == "inspect":
                _print(record)
            elif args.command == "ledger":
                _print(pd.read_parquet(record["ledger_path"]).to_dict(orient="records"))
            else:
                _print(json.loads(Path(record["summary_path"]).read_text(encoding="utf-8")))
            return 0
        if args.group == "data" and args.command == "catalog":
            _print(catalog.datasets()); return 0
        if args.group=="data" and args.command in {"import-file","import-status","import-inspect","import-list","certify"}:
            gateway=ExternalImportGateway(catalog,settings.data_dir)
            if args.command=="import-file":
                keys=("timestamp","open","high","low","close") if args.resolution in {"M1","M15"} else ("timestamp","bid","ask")
                mapping={key:getattr(args,key+"_column") for key in keys}
                for key in ("tick_volume","real_volume","spread","bid_volume","ask_volume"):
                    value=getattr(args,key+"_column",None)
                    if value:mapping[key]=value
                spec=ImportSpec(tuple(args.file),args.provider,args.symbol,args.canonical_symbol or args.symbol,Resolution(args.resolution),PriceType(args.price_type),args.timezone,mapping,args.target_execution_broker,args.derive_mid,args.interpretation_reason,TimestampFormat(args.timestamp_format))
                value=gateway.import_files(spec)
            elif args.command=="import-list":value=gateway.list_imports()
            elif args.command=="certify":value=gateway.certify(args.dataset_id)
            else:value=gateway.inspect_dataset(args.import_id)
            _print(value);return 0
        if args.group == "data" and args.command == "audit":
            path = Path(args.target)
            if not catalog.path.exists():PartitionService(ResearchRegistry(settings.catalog_path))
            if path.exists():_assert_unprotected_market_path(catalog,path)
            paths = [path] if path.exists() else list((settings.data_dir / "raw").rglob(f"{args.target}/M1/**/*.parquet"))
            if not paths:
                raise ValueError(f"no Parquet datasets found for {args.target}")
            for p in paths:_assert_unprotected_market_path(catalog,p)
            frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
            report = audit_market_data(frame).to_dict(); catalog.add_integrity(None, report); _print(report); return 0 if report["status"] != "error" else 2
        adapter = _connect(settings)
        try:
            status = adapter.status()
            if args.group == "mt5":
                _print(status.__dict__); return 0
            symbol = adapter.resolve_symbol(args.pair)
            catalog.upsert_symbol(status.server or "unknown", symbol)
            if args.group == "symbols":
                _print(symbol.model_dump()); return 0
            if args.command == "inspect":
                result = inspect_history(adapter, symbol.broker_symbol)
                _print({"broker": status.company, "server": status.server, **result.__dict__}); return 0
            if args.command == "download":
                end = args.end or datetime.now(timezone.utc)
                start = args.start or end - timedelta(days=1)
                if start >= end:
                    raise ValueError("--start must be before --end")
                def progress(i, total, left, right, rows):
                    print(f"chunk {i}/{total}: {left.isoformat()} to {right.isoformat()} ({rows} rows)", file=sys.stderr)
                frame = download(adapter, symbol.broker_symbol, start, end, args.chunk_days or settings.chunk_days, settings.max_retries, progress)
                catalog.record_download(symbol.broker_symbol, start.isoformat(), end.isoformat(), "empty" if frame.empty else "retrieved", f"rows={len(frame)}")
                paths = RawStore(settings.data_dir,catalog).write_months(frame, status.server or "unknown", symbol.broker_symbol, status.company or "unknown", start, end)
                records = [catalog.dataset_for_path(path) for path in paths]
                _print(records); return 0
        finally:
            adapter.shutdown()
    except (MT5Error, ResearchError, ExternalImportError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
