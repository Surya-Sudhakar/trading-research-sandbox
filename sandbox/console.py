from __future__ import annotations

import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Callable

from sandbox.cli import main as cli_main

InputFn = Callable[[str], str]


def _prompt(label: str, input_fn: InputFn, optional: bool = False) -> str:
    while True:
        try:
            value = input_fn(f"{label}: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return ""
        if value or optional:
            return value
        print("Value required.")


def _pause(input_fn: InputFn) -> None:
    try:
        input_fn("\nPress Enter to continue...")
    except (EOFError, KeyboardInterrupt):
        pass


def _confirm(token: str, message: str, input_fn: InputFn) -> bool:
    print(message)
    try:
        return input_fn(f"Type {token} to continue: ").strip() == token
    except (EOFError, KeyboardInterrupt):
        return False


def _run(argv: list[str]) -> int:
    print("\n$ python -m sandbox " + " ".join(argv))
    print("-" * 72)
    try:
        code = cli_main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    except KeyboardInterrupt:
        print("\nCancelled.")
        code = 130
    print("-" * 72)
    print(f"Exit code: {code}")
    return code


def _menu(title: str, items: list[tuple[str, str]], input_fn: InputFn) -> str:
    while True:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for key, label in items:
            print(f" [{key}] {label}")
        print(" [0] Back")
        try:
            choice = input_fn("\nSelect > ").strip()
        except (EOFError, KeyboardInterrupt):
            return "0"
        if choice == "0" or any(choice == key for key, _ in items):
            return choice
        print("Invalid selection.")


def _program(command: str, input_fn: InputFn) -> None:
    program_id = _prompt("Program ID", input_fn)
    if program_id:
        _run(["research", command, program_id])


def overview(input_fn: InputFn) -> None:
    items = [("1", "MT5 status"), ("2", "Data catalog"), ("3", "Registered strategies"),
             ("4", "Research journal integrity"), ("5", "Program status"), ("6", "What next")]
    while True:
        c = _menu("OVERVIEW / SYSTEM STATUS", items, input_fn)
        if c == "0": return
        if c == "1": _run(["mt5", "status"])
        elif c == "2": _run(["data", "catalog"])
        elif c == "3": _run(["strategies", "list"])
        elif c == "4": _run(["research", "journal", "verify"])
        elif c == "5": _program("status", input_fn)
        elif c == "6": _program("next", input_fn)
        _pause(input_fn)


def data_menu(input_fn: InputFn) -> None:
    items = [("1", "Data catalog"), ("2", "External import list"), ("3", "Audit dataset / symbol"),
             ("4", "Inspect partition"), ("5", "Verify partition"), ("6", "Partition access history"),
             ("7", "Inspect broker history")]
    while True:
        c = _menu("MARKET DATA", items, input_fn)
        if c == "0": return
        if c == "1": _run(["data", "catalog"])
        elif c == "2": _run(["data", "import-list"])
        elif c == "3":
            target = _prompt("Parquet path or broker symbol", input_fn)
            if target: _run(["data", "audit", target])
        elif c in {"4", "5", "6"}:
            pid = _prompt("Partition ID", input_fn)
            cmd = {"4": "inspect", "5": "verify", "6": "access-history"}[c]
            if pid: _run(["data", "partition", cmd, pid])
        elif c == "7":
            pair = _prompt("Pair, e.g. EURUSD", input_fn)
            if pair: _run(["history", "inspect", pair])
        _pause(input_fn)



def _manual_proposal_wizard(input_fn: InputFn, *, execute_after: bool = False) -> None:
    root_text = _prompt("Discovery results root (blank = results/discovery_runner)", input_fn, True)
    root = Path(root_text or "results/discovery_runner")
    prepared_id = _prompt("Prepared ID", input_fn)
    if not prepared_id:
        return

    packet_path = root / "prepared" / prepared_id / "packet.json"
    if not packet_path.exists():
        print(f"Prepared packet not found: {packet_path}")
        return
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unable to read packet: {exc}")
        return

    allowed = tuple(packet.get("allowed_condition_feature_ids", ()))
    print("\nAllowed condition fields:")
    for field_id in allowed:
        print(f"  - {field_id}")
    print(f"\nTarget: {packet.get('target_field_id')}")
    print(f"Packet fingerprint: {packet.get('fingerprint')}")

    hypothesis_id = _prompt("Hypothesis ID (A-Z, 0-9, _)", input_fn)
    if not re.fullmatch(r"[A-Z0-9_]+", hypothesis_id or ""):
        print("Invalid hypothesis ID. Use only A-Z, 0-9, and _.")
        return
    statement = _prompt("Hypothesis statement", input_fn)
    rationale = _prompt("Rationale", input_fn)
    family_id = _prompt("Family ID (blank = MANUAL_DISCOVERY)", input_fn, True) or "MANUAL_DISCOVERY"
    strategy_id = _prompt("Strategy ID (blank = EURUSD_MANUAL)", input_fn, True) or "EURUSD_MANUAL"
    variant_id = _prompt("Variant ID (blank = V1)", input_fn, True) or "V1"

    conditions = []
    print("\nAdd one or more conditions. Leave field_id blank when finished.")
    while True:
        field_id = _prompt("Condition field_id", input_fn, True)
        if not field_id:
            break
        operator = _prompt("Operator [EQ/NE/GT/GTE/LT/LTE/IN/NOT_IN]", input_fn)
        raw_value = _prompt('Value as JSON (e.g. 0.70, true, or "TOUCH")', input_fn)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON value: {exc}")
            continue
        conditions.append({"field_id": field_id, "operator": operator, "value": value})

    if not conditions:
        print("At least one condition is required.")
        return

    side = _prompt("Side [LONG/SHORT]", input_fn).upper()
    if side not in {"LONG", "SHORT"}:
        print("Side must be LONG or SHORT.")
        return
    try:
        stop_pips = float(_prompt("Stop pips", input_fn))
        target_r = float(_prompt("Target R multiple", input_fn))
    except ValueError:
        print("Stop pips and target R must be numeric.")
        return

    proposal = {
        "version": "HYPOTHESIS_PROPOSAL_V1",
        "status": "UNVERIFIED_DISCOVERY_HYPOTHESIS",
        "research_packet_fingerprint": packet.get("fingerprint"),
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "rationale": rationale,
        "target_field_id": packet.get("target_field_id"),
        "family_id": family_id,
        "strategy_id": strategy_id,
        "variant_id": variant_id,
        "evidence_feature_ids": list(dict.fromkeys(item["field_id"] for item in conditions)),
        "rules": [{
            "rule_id": "RULE_1",
            "conditions": conditions,
            "condition_logic": "ALL",
            "side": side,
            "stop_pips": stop_pips,
            "target_r_multiple": target_r,
        }],
    }

    manual_dir = root / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = manual_dir / f"{hypothesis_id}.json"
    if proposal_path.exists():
        print(f"Proposal already exists and will not be overwritten: {proposal_path}")
        return
    proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nCreated proposal: {proposal_path}")

    if not execute_after:
        return
    pip_size = _prompt("Pip size (blank = 0.0001)", input_fn, True) or "0.0001"
    config = _prompt("Execution config JSON path (blank = auto M15 interval + defaults)", input_fn, True)
    if not _confirm("EXECUTE", "Execute this proposal on the prepared Discovery dataset only?", input_fn):
        print("Execution cancelled. Proposal file was kept.")
        return
    argv = ["research", "discovery", "execute", prepared_id, str(proposal_path),
            "--root", str(root), "--pip-size", pip_size]
    if config:
        argv += ["--config", config]
    _run(argv)


def discovery_manual_menu(input_fn: InputFn) -> None:
    items = [
        ("1", "Inspect prepared Discovery run"),
        ("2", "Create manual hypothesis file"),
        ("3", "Execute existing proposal file"),
        ("4", "Create + execute manual hypothesis"),
    ]
    while True:
        c = _menu("PREPARED DISCOVERY / MANUAL HYPOTHESIS", items, input_fn)
        if c == "0":
            return
        if c == "1":
            root = _prompt("Discovery results root (blank = results/discovery_runner)", input_fn, True) or "results/discovery_runner"
            prepared_id = _prompt("Prepared ID", input_fn)
            if prepared_id:
                _run(["research", "discovery", "inspect-prepared", prepared_id, "--root", root])
        elif c == "2":
            _manual_proposal_wizard(input_fn, execute_after=False)
        elif c == "3":
            root = _prompt("Discovery results root (blank = results/discovery_runner)", input_fn, True) or "results/discovery_runner"
            prepared_id = _prompt("Prepared ID", input_fn)
            proposal = _prompt("Proposal JSON path", input_fn)
            pip_size = _prompt("Pip size (blank = 0.0001)", input_fn, True) or "0.0001"
            config = _prompt("Execution config JSON path (blank = auto interval + defaults)", input_fn, True)
            if prepared_id and proposal and _confirm("EXECUTE", "Execute proposal on Discovery only?", input_fn):
                argv = ["research", "discovery", "execute", prepared_id, proposal,
                        "--root", root, "--pip-size", pip_size]
                if config:
                    argv += ["--config", config]
                _run(argv)
        elif c == "4":
            _manual_proposal_wizard(input_fn, execute_after=True)
        _pause(input_fn)


def research_menu(input_fn: InputFn) -> None:
    items = [("1", "Program status"), ("2", "Question tree"), ("3", "What next"), ("4", "Timeline"),
             ("5", "Inspect question"), ("6", "Approve hypothesis"), ("7", "Verify journal"),
             ("8", "Prepared Discovery / Manual Hypothesis")]
    while True:
        c = _menu("MARKET RESEARCH / RESEARCH LAB", items, input_fn)
        if c == "0": return
        if c in {"1", "2", "3", "4"}:
            _program({"1": "status", "2": "tree", "3": "next", "4": "timeline"}[c], input_fn)
        elif c == "5":
            q = _prompt("Question ID", input_fn)
            if q: _run(["research", "question", "inspect", q])
        elif c == "6":
            h = _prompt("Hypothesis ID", input_fn)
            if h: _run(["research", "hypothesis", "approve", h])
        elif c == "7":
            _run(["research", "journal", "verify"])
        else:
            discovery_manual_menu(input_fn)
        _pause(input_fn)


def strategies_menu(input_fn: InputFn) -> None:
    items = [("1", "List strategies"), ("2", "Inspect strategy"), ("3", "Validate strategy plugin"),
             ("4", "Strategy fingerprint"), ("5", "Inspect research family"), ("6", "Family parameters")]
    while True:
        c = _menu("STRATEGIES", items, input_fn)
        if c == "0": return
        if c == "1": _run(["strategies", "list"])
        elif c in {"2", "3", "4"}:
            sid = _prompt("Strategy ID", input_fn)
            ver = _prompt("Version (blank = default/latest)", input_fn, True)
            argv = ["strategies", {"2": "inspect", "3": "validate", "4": "fingerprint"}[c], sid]
            if ver: argv += ["--version", ver]
            if sid: _run(argv)
        else:
            fid = _prompt("Family ID", input_fn)
            if fid: _run(["research", "family", "inspect" if c == "5" else "parameters", fid])
        _pause(input_fn)

def backtests_menu(input_fn: InputFn) -> None:
    items = [("1", "Run Discovery backtest"), ("2", "Inspect backtest"), ("3", "Backtest summary"),
             ("4", "Backtest ledger"), ("5", "Analyze run statistics"), ("6", "Inspect statistics report")]
    while True:
        c = _menu("EXPERIMENTS / BACKTESTS", items, input_fn)
        if c == "0": return
        if c == "1":
            dataset = _prompt("Market Parquet path", input_fn)
            intents = _prompt("TradeIntent JSON path", input_fn)
            config = _prompt("Execution config JSON path (blank = defaults)", input_fn, True)
            results = _prompt("Results directory (blank = results)", input_fn, True)
            argv = ["backtest", "run", dataset, intents]
            if config: argv += ["--config", config]
            if results: argv += ["--results-dir", results]
            if dataset and intents: _run(argv)
        elif c in {"2", "3", "4"}:
            run_id = _prompt("Run ID", input_fn)
            if run_id: _run(["backtest", {"2": "inspect", "3": "summary", "4": "ledger"}[c], run_id])
        elif c == "5":
            run_id = _prompt("Run ID", input_fn)
            mode = _prompt("Mode [DISCOVERY/VALIDATION/FINAL_TEST] (blank = DISCOVERY)", input_fn, True) or "DISCOVERY"
            if run_id: _run(["stats", "analyze", run_id, "--mode", mode])
        else:
            rid = _prompt("Report ID", input_fn)
            if rid: _run(["stats", "inspect", rid])
        _pause(input_fn)


def candidates_menu(input_fn: InputFn) -> None:
    items = [("1", "Inspect candidate"), ("2", "Candidate card"), ("3", "Candidate lineage"),
             ("4", "Candidate burden"), ("5", "Freeze candidate"), ("6", "Request validation"),
             ("7", "Inspect contamination")]
    while True:
        c = _menu("CANDIDATES", items, input_fn)
        if c == "0": return
        cid = _prompt("Candidate ID", input_fn)
        if not cid:
            _pause(input_fn); continue
        if c == "5":
            datasets = _prompt("Validation dataset IDs (comma separated)", input_fn)
            if datasets and _confirm("FREEZE", "Freeze only after Discovery evidence is complete.", input_fn):
                argv = ["research", "candidate", "freeze", cid]
                for ds in (x.strip() for x in datasets.split(",")):
                    if ds: argv += ["--validation-dataset", ds]
                _run(argv)
        elif c == "7": _run(["research", "contamination", "inspect", cid])
        else:
            cmd = {"1": "inspect", "2": "card", "3": "lineage", "4": "burden", "6": "validate"}[c]
            _run(["research", "candidate", cmd, cid])
        _pause(input_fn)


def validation_menu(input_fn: InputFn) -> None:
    items = [("1", "Bind frozen candidate to Validation partition"), ("2", "Run sealed Validation")]
    while True:
        c = _menu("VALIDATION - PROTECTED", items, input_fn)
        if c == "0": return
        cid = _prompt("Candidate ID", input_fn)
        pid = _prompt("Validation partition ID", input_fn)
        if c == "1":
            if cid and pid: _run(["research", "validation", "bind", cid, pid])
        else:
            intents = _prompt("Validation TradeIntent JSON path", input_fn)
            if cid and pid and intents and _confirm("VALIDATE", "Invoke sealed Validation using existing safeguards?", input_fn):
                _run(["research", "validation", "run", cid, pid, intents])
        _pause(input_fn)


def final_menu(input_fn: InputFn) -> None:
    items = [("1", "Authorize Final Test"), ("2", "Run sealed Final Test")]
    while True:
        c = _menu("FINAL TEST - PROTECTED / ONE-SHOT", items, input_fn)
        if c == "0": return
        cid = _prompt("Candidate ID", input_fn)
        pid = _prompt("Final-Test partition ID", input_fn)
        if c == "1":
            vid = _prompt("Validation result ID", input_fn)
            by = _prompt("Authorized by", input_fn)
            if all((cid, pid, vid, by)) and _confirm("FINAL", "Authorize protected Final Test?", input_fn):
                _run(["research", "final", "authorize", cid, pid, vid, "--authorized-by", by])
        else:
            intents = _prompt("Final-Test TradeIntent JSON path", input_fn)
            if cid and pid and intents and _confirm("FINAL", "Final Test may be one-shot. Continue?", input_fn):
                _run(["research", "final", "run", cid, pid, intents])
        _pause(input_fn)


def audit_menu(input_fn: InputFn) -> None:
    items = [("1", "Pre-run research audit"), ("2", "Post-run research audit"),
             ("3", "Audit history"), ("4", "Inspect audit"), ("5", "Inspect candidate contamination")]
    while True:
        c = _menu("AUDIT / CONTAMINATION", items, input_fn)
        if c == "0": return
        if c in {"1", "2", "3"}:
            eid = _prompt("Experiment ID", input_fn)
            if eid: _run(["research", "audit", {"1": "pre", "2": "post", "3": "history"}[c], eid])
        elif c == "4":
            aid = _prompt("Audit ID", input_fn)
            if aid: _run(["research", "audit", "inspect", aid])
        else:
            cid = _prompt("Candidate ID", input_fn)
            if cid: _run(["research", "contamination", "inspect", cid])
        _pause(input_fn)


def environment(input_fn: InputFn) -> None:
    print("\n" + "=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Executable  : {sys.executable}")
    print(f"Platform    : {platform.platform()}")
    print(f"Working dir : {Path.cwd()}")
    print(f"Process ID  : {os.getpid()}")
    print("\nThis menu is only a control surface. Existing sandbox services remain authoritative.")
    _pause(input_fn)


def run_console(*, input_fn: InputFn = input) -> int:
    actions = {"1": overview, "2": data_menu, "3": research_menu, "4": strategies_menu,
               "5": backtests_menu, "6": candidates_menu, "7": validation_menu,
               "8": final_menu, "9": audit_menu, "10": environment}
    while True:
        print("\n" + "=" * 72)
        print(" AUTONOMOUS TRADING R&D SYSTEM")
        print(" Interactive Research Console")
        print("=" * 72)
        print(" Existing CLI + backend remain authoritative\n")
        print(" [1]  Overview / System Status")
        print(" [2]  Market Data")
        print(" [3]  Market Research / Research Lab")
        print(" [4]  Strategies")
        print(" [5]  Experiments / Backtests")
        print(" [6]  Candidates")
        print(" [7]  Validation")
        print(" [8]  Final Test")
        print(" [9]  Audit / Contamination")
        print(" [10] Environment")
        print("\n [0]  Exit")
        try:
            choice = input_fn("\nSelect > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nResearch console closed.")
            return 0
        if choice == "0":
            print("Research console closed.")
            return 0
        action = actions.get(choice)
        if action is None:
            print("Invalid selection.")
            continue
        try:
            action(input_fn)
        except EOFError:
            print("\nEOF received. Returning to main menu.")
        except KeyboardInterrupt:
            print("\nCancelled. Returning to main menu.")


if __name__ == "__main__":
    raise SystemExit(run_console())
