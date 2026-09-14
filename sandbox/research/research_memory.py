from __future__ import annotations

"""Append-only research memory for pipeline runs and future Phase-2 candidates.

Git keeps code/configuration and compact manifests. DuckDB keeps the growing
scientific memory: what was run, what was tried, what failed, and why.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import duckdb

from sandbox.research.canonical import canonical_json, sha256_canonical


MEMORY_SCHEMA_VERSION = "SANDBOX_RESEARCH_MEMORY_V1"
_CANDIDATE_VERDICTS = {"PROMISING", "WEAK", "REJECTED", "SUPPORTED", "INCONCLUSIVE"}


DDL = """
CREATE TABLE IF NOT EXISTS memory_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    record_fingerprint VARCHAR NOT NULL UNIQUE,
    experiment_id VARCHAR NOT NULL,
    hypothesis VARCHAR NOT NULL,
    pipeline_version VARCHAR NOT NULL,
    git_commit VARCHAR NOT NULL,
    partition_id VARCHAR NOT NULL,
    partition_fingerprint VARCHAR NOT NULL,
    source_dataset_id VARCHAR NOT NULL,
    source_dataset_checksum VARCHAR NOT NULL,
    spec_fingerprint VARCHAR NOT NULL,
    metrics_json VARCHAR NOT NULL,
    record_json VARCHAR NOT NULL,
    remembered_utc TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_experiment ON pipeline_runs(experiment_id);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_dataset ON pipeline_runs(source_dataset_checksum);

CREATE TABLE IF NOT EXISTS candidate_results (
    candidate_id VARCHAR PRIMARY KEY,
    candidate_fingerprint VARCHAR NOT NULL UNIQUE,
    hypothesis_id VARCHAR,
    parent_candidate_id VARCHAR,
    experiment_id VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL,
    parameters_json VARCHAR NOT NULL,
    metrics_json VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    rejection_reason VARCHAR,
    record_fingerprint VARCHAR NOT NULL UNIQUE,
    record_json VARCHAR NOT NULL,
    remembered_utc TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_candidate_results_hypothesis ON candidate_results(hypothesis_id);
CREATE INDEX IF NOT EXISTS ix_candidate_results_parent ON candidate_results(parent_candidate_id);
CREATE INDEX IF NOT EXISTS ix_candidate_results_verdict ON candidate_results(verdict);
CREATE INDEX IF NOT EXISTS ix_candidate_results_rejection ON candidate_results(rejection_reason);
"""


@dataclass(frozen=True)
class RememberResult:
    inserted: bool
    record_fingerprint: str


class ResearchMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        return duckdb.connect(str(self.path))

    def initialize(self) -> None:
        with self.connect() as con:
            con.execute(DDL)
            row = con.execute("SELECT value FROM memory_meta WHERE key='schema_version'").fetchone()
            if row is None:
                con.execute("INSERT INTO memory_meta VALUES ('schema_version', ?)", [MEMORY_SCHEMA_VERSION])
            elif row[0] != MEMORY_SCHEMA_VERSION:
                raise ValueError(f"unsupported research memory schema: {row[0]}")

    @staticmethod
    def _payload_fingerprint(payload: dict[str, Any]) -> str:
        return sha256_canonical(payload)

    def remember_pipeline_run(
        self,
        *,
        manifest: dict[str, Any],
        metrics: dict[str, Any],
        hypothesis: str,
    ) -> RememberResult:
        required = (
            "run_id", "experiment_id", "pipeline_version", "git_commit", "partition_id",
            "partition_fingerprint", "source_dataset_id", "source_dataset_checksum", "spec_fingerprint",
        )
        missing = [name for name in required if not manifest.get(name)]
        if missing:
            raise ValueError("pipeline manifest missing: " + ", ".join(missing))
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("hypothesis must be nonempty")
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")

        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "experiment_id": manifest["experiment_id"],
            "hypothesis": hypothesis,
            "pipeline_version": manifest["pipeline_version"],
            "git_commit": manifest["git_commit"],
            "partition_id": manifest["partition_id"],
            "partition_fingerprint": manifest["partition_fingerprint"],
            "source_dataset_id": manifest["source_dataset_id"],
            "source_dataset_checksum": manifest["source_dataset_checksum"],
            "spec_fingerprint": manifest["spec_fingerprint"],
            "metrics": metrics,
        }
        fingerprint = self._payload_fingerprint(payload)
        encoded = canonical_json(payload)
        metrics_json = canonical_json(metrics)
        with self.connect() as con:
            existing = con.execute(
                "SELECT record_fingerprint FROM pipeline_runs WHERE run_id=?", [manifest["run_id"]]
            ).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise ValueError("immutable research run conflict")
                return RememberResult(False, fingerprint)
            con.execute(
                """INSERT INTO pipeline_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    manifest["run_id"], fingerprint, manifest["experiment_id"], hypothesis,
                    manifest["pipeline_version"], manifest["git_commit"], manifest["partition_id"],
                    manifest["partition_fingerprint"], manifest["source_dataset_id"],
                    manifest["source_dataset_checksum"], manifest["spec_fingerprint"], metrics_json,
                    encoded, datetime.now(timezone.utc),
                ],
            )
        return RememberResult(True, fingerprint)

    def remember_candidate(
        self,
        *,
        candidate_id: str,
        candidate_fingerprint: str,
        experiment_id: str,
        run_id: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        verdict: str,
        rejection_reason: str | None = None,
        hypothesis_id: str | None = None,
        parent_candidate_id: str | None = None,
    ) -> RememberResult:
        for name, value in (
            ("candidate_id", candidate_id), ("candidate_fingerprint", candidate_fingerprint),
            ("experiment_id", experiment_id), ("run_id", run_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        verdict = verdict.upper()
        if verdict not in _CANDIDATE_VERDICTS:
            raise ValueError(f"unsupported candidate verdict: {verdict}")
        if verdict == "REJECTED" and (not isinstance(rejection_reason, str) or not rejection_reason.strip()):
            raise ValueError("REJECTED candidate requires rejection_reason")
        if not isinstance(parameters, dict) or not isinstance(metrics, dict):
            raise TypeError("parameters and metrics must be dictionaries")

        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "candidate_fingerprint": candidate_fingerprint,
            "hypothesis_id": hypothesis_id,
            "parent_candidate_id": parent_candidate_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "parameters": parameters,
            "metrics": metrics,
            "verdict": verdict,
            "rejection_reason": rejection_reason,
        }
        fingerprint = self._payload_fingerprint(payload)
        encoded = canonical_json(payload)
        with self.connect() as con:
            existing = con.execute(
                "SELECT record_fingerprint FROM candidate_results WHERE candidate_id=?", [candidate_id]
            ).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise ValueError("immutable candidate memory conflict")
                return RememberResult(False, fingerprint)
            duplicate = con.execute(
                "SELECT candidate_id FROM candidate_results WHERE candidate_fingerprint=?",
                [candidate_fingerprint],
            ).fetchone()
            if duplicate:
                raise ValueError(f"duplicate candidate fingerprint already stored as {duplicate[0]}")
            con.execute(
                """INSERT INTO candidate_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    candidate_id, candidate_fingerprint, hypothesis_id, parent_candidate_id,
                    experiment_id, run_id, canonical_json(parameters), canonical_json(metrics), verdict,
                    rejection_reason, fingerprint, encoded, datetime.now(timezone.utc),
                ],
            )
        return RememberResult(True, fingerprint)

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT record_json FROM candidate_results WHERE candidate_id=?", [candidate_id]
            ).fetchone()
        return json.loads(row[0]) if row else None

    def candidate_exists(self, candidate_fingerprint: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT 1 FROM candidate_results WHERE candidate_fingerprint=?", [candidate_fingerprint]
            ).fetchone()
        return row is not None

    def children_of(self, parent_candidate_id: str) -> tuple[dict[str, Any], ...]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT record_json FROM candidate_results WHERE parent_candidate_id=? ORDER BY candidate_id",
                [parent_candidate_id],
            ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def failure_summary(self) -> tuple[dict[str, Any], ...]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT rejection_reason, count(*) AS candidate_count
                   FROM candidate_results
                   WHERE verdict='REJECTED'
                   GROUP BY rejection_reason
                   ORDER BY candidate_count DESC, rejection_reason"""
            ).fetchall()
        return tuple({"rejection_reason": row[0], "candidate_count": row[1]} for row in rows)

    def counts(self) -> dict[str, int]:
        with self.connect() as con:
            runs = con.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0]
            candidates = con.execute("SELECT count(*) FROM candidate_results").fetchone()[0]
            rejected = con.execute("SELECT count(*) FROM candidate_results WHERE verdict='REJECTED'").fetchone()[0]
        return {"pipeline_runs": runs, "candidates": candidates, "rejected_candidates": rejected}
