from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sandbox.execution.engine import RunResult
from sandbox.execution.metrics import summarize, symbol_breakdown, yearly_breakdown
from sandbox.provenance import sha256_file


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    run_fingerprint: str
    ledger_path: Path
    summary_path: Path
    ledger_checksum: str


def store_run(result: RunResult, root: Path = Path("results")) -> StoredRun:
    run_id = "run-" + result.run_fingerprint[:20]
    folder = root / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledgers" / f"{run_id}.parquet"
    summary_path = root / "summaries" / f"{run_id}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.frame()
    if ledger_path.exists():
        existing = pd.read_parquet(ledger_path)
        pd.testing.assert_frame_equal(existing, frame, check_dtype=False)
    else:
        frame.to_parquet(ledger_path, index=False)
    summary = {
        "run_id": run_id, "run_fingerprint": result.run_fingerprint,
        "execution_engine_version": result.execution_engine_version,
        "ledger_schema_version": result.ledger_schema_version, "metrics_version": result.metrics_version,
        "execution_config": result.execution_config.model_dump(mode="json"),
        "metrics": summarize(frame), "yearly_breakdown": yearly_breakdown(frame),
        "symbol_breakdown": symbol_breakdown(frame), "ledger_checksum": sha256_file(ledger_path),
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != serialized:
        raise ValueError("deterministic run summary differs from existing artifact")
    summary_path.write_text(serialized, encoding="utf-8")
    (folder / "execution_config.json").write_text(json.dumps(result.execution_config.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return StoredRun(run_id, result.run_fingerprint, ledger_path, summary_path, summary["ledger_checksum"])

