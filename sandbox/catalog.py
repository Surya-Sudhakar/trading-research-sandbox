from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from sandbox import SCHEMA_VERSION, __version__
from sandbox.models import SymbolRecord
from sandbox.provenance import Provenance


DDL = """
CREATE TABLE IF NOT EXISTS app_versions (version TEXT NOT NULL, schema_version TEXT NOT NULL, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS symbols (server TEXT NOT NULL, broker_symbol TEXT NOT NULL, canonical_pair TEXT, metadata_json TEXT NOT NULL, discovered_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(server, broker_symbol));
CREATE TABLE IF NOT EXISTS downloads (id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, requested_start TEXT NOT NULL, requested_end TEXT NOT NULL, status TEXT NOT NULL, detail TEXT, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS datasets (dataset_id TEXT PRIMARY KEY, broker TEXT NOT NULL, server TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL, row_count INTEGER NOT NULL, path TEXT NOT NULL UNIQUE, checksum TEXT NOT NULL, schema_version TEXT NOT NULL, retrieved_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS provenance (dataset_id TEXT PRIMARY KEY REFERENCES datasets(dataset_id), record_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dataset_provenance_seals (dataset_id TEXT PRIMARY KEY REFERENCES datasets(dataset_id), provenance_fingerprint TEXT NOT NULL, sealed_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS integrity_reports (id INTEGER PRIMARY KEY, dataset_id TEXT, status TEXT NOT NULL, report_json TEXT NOT NULL, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS research_runs (run_id TEXT PRIMARY KEY, run_fingerprint TEXT NOT NULL UNIQUE, dataset_id TEXT NOT NULL, engine_version TEXT NOT NULL, ledger_schema_version TEXT NOT NULL, metrics_version TEXT NOT NULL, execution_config TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, ledger_path TEXT NOT NULL, summary_path TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS external_imports (import_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL UNIQUE, scientific_fingerprint TEXT NOT NULL UNIQUE, status TEXT NOT NULL, manifest_json TEXT NOT NULL, manifest_fingerprint TEXT NOT NULL, audit_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS external_import_components (import_id TEXT NOT NULL REFERENCES external_imports(import_id), ordinal INTEGER NOT NULL, original_path TEXT NOT NULL, preserved_path TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, PRIMARY KEY(import_id, ordinal));
CREATE TABLE IF NOT EXISTS dataset_classifications(dataset_id TEXT NOT NULL, classification TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(dataset_id,classification));
"""


class Catalog:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA foreign_keys=ON")
            yield con
            con.commit()
        finally:
            con.close()

    def initialize(self) -> None:
        with self.connection() as con:
            con.executescript(DDL)
            found = con.execute("SELECT 1 FROM app_versions WHERE version=? AND schema_version=?", (__version__, SCHEMA_VERSION)).fetchone()
            if not found:
                con.execute("INSERT INTO app_versions(version, schema_version) VALUES (?,?)", (__version__, SCHEMA_VERSION))
            for row in con.execute("SELECT p.dataset_id,p.record_json FROM provenance p LEFT JOIN dataset_provenance_seals s ON s.dataset_id=p.dataset_id WHERE s.dataset_id IS NULL"):
                con.execute("INSERT INTO dataset_provenance_seals(dataset_id,provenance_fingerprint) VALUES (?,?)",(row[0],hashlib.sha256(row[1].encode("utf-8")).hexdigest()))

    def upsert_symbol(self, server: str, symbol: SymbolRecord) -> None:
        self.initialize()
        with self.connection() as con:
            con.execute("INSERT INTO symbols(server,broker_symbol,canonical_pair,metadata_json) VALUES (?,?,?,?) ON CONFLICT(server,broker_symbol) DO UPDATE SET canonical_pair=excluded.canonical_pair, metadata_json=excluded.metadata_json, discovered_at=CURRENT_TIMESTAMP", (server, symbol.broker_symbol, symbol.canonical_pair, symbol.model_dump_json()))

    def record_download(self, symbol: str, start: str, end: str, status: str, detail: str = "") -> None:
        self.initialize()
        with self.connection() as con:
            con.execute("INSERT INTO downloads(symbol,requested_start,requested_end,status,detail) VALUES (?,?,?,?,?)", (symbol, start, end, status, detail))

    def add_dataset(self, provenance: Provenance) -> None:
        self.initialize()
        p = provenance
        with self.connection() as con:
            con.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (p.dataset_id,p.broker,p.server,p.symbol,p.timeframe,p.start,p.end,p.row_count,p.path,p.checksum,p.schema_version,p.retrieved_at))
            record=json.dumps(p.to_dict(), sort_keys=True);con.execute("INSERT INTO provenance(dataset_id,record_json) VALUES (?,?)", (p.dataset_id,record));con.execute("INSERT INTO dataset_provenance_seals(dataset_id,provenance_fingerprint) VALUES (?,?)",(p.dataset_id,hashlib.sha256(record.encode("utf-8")).hexdigest()))

    def add_integrity(self, dataset_id: str | None, report: dict) -> None:
        self.initialize()
        with self.connection() as con:
            con.execute("INSERT INTO integrity_reports(dataset_id,status,report_json) VALUES (?,?,?)", (dataset_id, report["status"], json.dumps(report, sort_keys=True)))

    def datasets(self) -> list[dict]:
        self.initialize()
        with self.connection() as con:
            return [dict(row) for row in con.execute("SELECT * FROM datasets ORDER BY retrieved_at, dataset_id")]

    def add_research_run(self, run_id: str, fingerprint: str, dataset_id: str, engine_version: str, ledger_schema_version: str, metrics_version: str, config: dict, ledger_path: str, summary_path: str, status: str = "COMPLETE") -> None:
        self.initialize()
        with self.connection() as con:
            con.execute("INSERT INTO research_runs(run_id,run_fingerprint,dataset_id,engine_version,ledger_schema_version,metrics_version,execution_config,ledger_path,summary_path,status) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status", (run_id, fingerprint, dataset_id, engine_version, ledger_schema_version, metrics_version, json.dumps(config, sort_keys=True), ledger_path, summary_path, status))

    def research_run(self, run_id: str) -> dict | None:
        self.initialize()
        with self.connection() as con:
            row = con.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None
