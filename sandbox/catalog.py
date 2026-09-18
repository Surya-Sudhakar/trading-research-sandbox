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
CREATE TABLE IF NOT EXISTS raw_dataset_revisions (dataset_id TEXT NOT NULL, revision INTEGER NOT NULL, checksum TEXT NOT NULL, provenance_json TEXT NOT NULL, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(dataset_id,revision), UNIQUE(dataset_id,checksum));
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

    def dataset_for_path(self, path: Path) -> dict | None:
        self.initialize()
        with self.connection() as con:
            row=self._raw_path_row(con,path)
            return dict(row) if row else None

    @staticmethod
    def _raw_path_row(con, path):
        matches=[row for row in con.execute("SELECT * FROM datasets")
                 if Path(row["path"]).resolve()==Path(path).resolve()]
        if len(matches)>1:
            raise ValueError("multiple catalog datasets resolve to the same raw path")
        return matches[0] if matches else None

    def upsert_raw_dataset(self, provenance: Provenance) -> Provenance:
        """Reconcile one canonical monthly raw artifact while retaining revision evidence."""
        self.initialize()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            existing=self._raw_path_row(con,provenance.path)
            dataset_id=existing["dataset_id"] if existing else provenance.dataset_id
            current=Provenance(dataset_id,provenance.broker,provenance.server,provenance.symbol,
                provenance.timeframe,provenance.start,provenance.end,provenance.retrieved_at,
                provenance.row_count,provenance.schema_version,provenance.checksum,
                existing["path"] if existing else provenance.path)
            if existing:
                old=con.execute("SELECT record_json FROM provenance WHERE dataset_id=?",(dataset_id,)).fetchone()
                seal=con.execute("SELECT provenance_fingerprint FROM dataset_provenance_seals WHERE dataset_id=?",(dataset_id,)).fetchone()
                if (not old or not seal or hashlib.sha256(old[0].encode("utf-8")).hexdigest()!=seal[0]
                        or any(str(json.loads(old[0]).get(key))!=str(existing[key]) for key in Provenance.__dataclass_fields__)):
                    raise ValueError("raw dataset provenance/catalog seal mismatch")
                if any(existing[key]!=getattr(current,key) for key in ("broker","server","symbol","timeframe")):
                    raise ValueError("raw dataset identity metadata mismatch")
                if old:
                    con.execute("INSERT OR IGNORE INTO raw_dataset_revisions(dataset_id,revision,checksum,provenance_json) VALUES (?,coalesce((SELECT max(revision)+1 FROM raw_dataset_revisions WHERE dataset_id=?),1),?,?)",
                                (dataset_id,dataset_id,existing["checksum"],old[0]))
                if existing["checksum"]==current.checksum and existing["row_count"]==current.row_count and existing["start"]==current.start and existing["end"]==current.end:
                    return Provenance(**{key:existing[key] for key in Provenance.__dataclass_fields__})
                con.execute("UPDATE datasets SET broker=?,server=?,symbol=?,timeframe=?,start=?,end=?,row_count=?,checksum=?,schema_version=?,retrieved_at=? WHERE dataset_id=?",
                            (current.broker,current.server,current.symbol,current.timeframe,current.start,current.end,current.row_count,current.checksum,current.schema_version,current.retrieved_at,dataset_id))
            else:
                con.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (current.dataset_id,current.broker,current.server,current.symbol,current.timeframe,current.start,current.end,current.row_count,current.path,current.checksum,current.schema_version,current.retrieved_at))
            record=json.dumps(current.to_dict(),sort_keys=True);seal=hashlib.sha256(record.encode("utf-8")).hexdigest()
            con.execute("INSERT INTO provenance(dataset_id,record_json) VALUES (?,?) ON CONFLICT(dataset_id) DO UPDATE SET record_json=excluded.record_json",(dataset_id,record))
            con.execute("INSERT INTO dataset_provenance_seals(dataset_id,provenance_fingerprint) VALUES (?,?) ON CONFLICT(dataset_id) DO UPDATE SET provenance_fingerprint=excluded.provenance_fingerprint,sealed_at=CURRENT_TIMESTAMP",(dataset_id,seal))
            con.execute("INSERT OR IGNORE INTO raw_dataset_revisions(dataset_id,revision,checksum,provenance_json) VALUES (?,coalesce((SELECT max(revision)+1 FROM raw_dataset_revisions WHERE dataset_id=?),1),?,?)",
                        (dataset_id,dataset_id,current.checksum,record))
            return current

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
