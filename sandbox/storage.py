from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd

from sandbox import SCHEMA_VERSION, __version__
from sandbox.integrity import audit
from sandbox.market_data.gaps import ContinuityPolicy, inventory
from sandbox.provenance import Provenance, sha256_file


COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


class ObservationConflictError(ValueError):
    def __init__(self, conflicts: pd.DataFrame, evidence_path: Path | None = None):
        self.conflicts = conflicts.copy()
        self.evidence_path = evidence_path
        timestamps = sorted(set(pd.to_datetime(conflicts["timestamp_utc"], utc=True).astype(str)))
        detail = f"conflicting observations at timestamps: {', '.join(timestamps)}"
        if evidence_path is not None:
            detail += f"; evidence preserved at {evidence_path}"
        super().__init__(detail)


def safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _merge_chunks(chunks: list[pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    tagged = [frame[COLUMNS].assign(observation_source=label)
              for frame, label in zip(chunks, labels) if not frame.empty]
    if not tagged:
        return pd.DataFrame(columns=COLUMNS)
    result = pd.concat(tagged, ignore_index=True)
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    conflicts = []
    duplicate_rows = result[result.timestamp_utc.duplicated(keep=False)]
    for _, group in duplicate_rows.groupby("timestamp_utc", sort=True):
        if len(group[COLUMNS[1:]].drop_duplicates()) > 1:
            conflicts.append(group)
    if conflicts:
        raise ObservationConflictError(pd.concat(conflicts, ignore_index=True))
    return (result.sort_values("timestamp_utc", kind="mergesort")
            .drop_duplicates("timestamp_utc", keep="first")[COLUMNS].reset_index(drop=True))


def normalize_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    return _merge_chunks(chunks, [f"chunk_{index}" for index in range(len(chunks))])


class RawStore:
    def __init__(self, data_dir: Path, catalog=None):
        self.data_dir = data_dir
        self.catalog = catalog

    def target(self, server: str, symbol: str, month: pd.Timestamp) -> Path:
        return self.data_dir / "raw" / "ic_markets" / safe_segment(server) / safe_segment(symbol) / "M1" / f"{month.year:04d}" / f"{month.month:02d}.parquet"

    def _preserve_conflict(self, conflicts: pd.DataFrame, server: str, symbol: str, month: pd.Timestamp) -> Path:
        payload = conflicts.to_json(orient="records", date_format="iso", double_precision=15)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path = (self.data_dir / "quarantine" / "conflicts" / safe_segment(server)
                / safe_segment(symbol) / f"{month.year:04d}" / f"{month.month:02d}-{fingerprint[:16]}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            conflicts.to_parquet(path, index=False)
        return path

    @staticmethod
    def _atomic_metadata(path: Path, metadata: dict) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            # Preserve the prior byte-level revision and a receipt before publish.
            # A crash after rename can then be distinguished from file tampering.
            previous_checksum = sha256_file(path) if path.exists() else None
            if previous_checksum is not None:
                revision = path.parent / ".revisions" / path.stem / (previous_checksum + ".parquet")
                revision.parent.mkdir(parents=True, exist_ok=True)
                if not revision.exists():
                    shutil.copyfile(path, revision)
                if sha256_file(revision) != previous_checksum:
                    raise ValueError("raw revision checksum mismatch")
                sidecar = path.with_suffix(".metadata.json")
                if sidecar.exists() and not revision.with_suffix(".metadata.json").exists():
                    shutil.copyfile(sidecar, revision.with_suffix(".metadata.json"))
            RawStore._atomic_metadata(path.with_suffix(".pending.json"), {
                "previous_checksum": previous_checksum, "checksum": sha256_file(temporary),
            })
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def write_months(self, frame: pd.DataFrame, server: str, symbol: str, broker: str,
                     requested_start: datetime, requested_end: datetime) -> list[Path]:
        if frame.empty:
            return []
        incoming = normalize_chunks([frame])
        report = audit(incoming)
        if any(x.category == "confirmed_data_error" for x in report.issues):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = self.data_dir / "quarantine" / safe_segment(server) / safe_segment(symbol) / f"{stamp}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            incoming.to_parquet(path, index=False)
            raise ValueError(f"invalid data quarantined at {path}")
        outputs = []
        groups = incoming.groupby([incoming.timestamp_utc.dt.year, incoming.timestamp_utc.dt.month])
        for (year, month_number), group in groups:
            month = pd.Timestamp(year=int(year), month=int(month_number), day=1, tz="UTC")
            path = self.target(server, symbol, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            registered = self.catalog.dataset_for_path(path) if self.catalog is not None else None
            pending_path = path.with_suffix(".pending.json")
            if registered:
                if not path.exists():
                    raise ValueError("registered raw artifact is missing")
                checksum = sha256_file(path)
                if checksum != registered["checksum"]:
                    receipt = json.loads(pending_path.read_text()) if pending_path.exists() else {}
                    if (receipt.get("previous_checksum") != registered["checksum"]
                            or receipt.get("checksum") != checksum):
                        raise ValueError("raw artifact checksum mismatch without recovery receipt")
                    # Complete an interrupted month before accepting further data.
                    restored = pd.read_parquet(path)
                    self.catalog.upsert_raw_dataset(Provenance.create(
                        path, broker, server, symbol, "M1",
                        restored.timestamp_utc.min(), restored.timestamp_utc.max(),
                        len(restored), SCHEMA_VERSION))
            existing = normalize_chunks([pd.read_parquet(path)]) if path.exists() else pd.DataFrame(columns=COLUMNS)
            try:
                table = _merge_chunks([existing, group], ["existing", "incoming"])
            except ObservationConflictError as exc:
                evidence = self._preserve_conflict(exc.conflicts, server, symbol, month)
                raise ObservationConflictError(exc.conflicts, evidence) from exc
            table_report = audit(table)
            if table_report.status == "error":
                raise ValueError(f"merged raw month failed integrity: {path}")
            content_changed = not path.exists() or not existing.equals(table)
            metadata_path = path.with_suffix(".metadata.json")
            try:
                prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                prior_metadata = {}
            first_timestamp = table.timestamp_utc.iloc[0]
            last_timestamp = table.timestamp_utc.iloc[-1]
            gaps, segments = inventory(table, SimpleNamespace(provider=broker, symbol=symbol, timeframe="M1"))
            now = datetime.now(timezone.utc).isoformat()
            metadata = {
                "broker": broker, "server": server, "broker_symbol": symbol, "timeframe": "M1",
                "retrieved_at": now if content_changed else prior_metadata.get("retrieved_at", now),
                "requested_start": requested_start.isoformat() if content_changed else prior_metadata.get("requested_start", requested_start.isoformat()),
                "requested_end": requested_end.isoformat() if content_changed else prior_metadata.get("requested_end", requested_end.isoformat()),
                "actual_start": first_timestamp.isoformat(), "actual_end": last_timestamp.isoformat(),
                "row_count": len(table), "software_version": __version__, "schema_version": SCHEMA_VERSION,
                "integrity_report": table_report.to_dict(),
                "continuity_policy": ContinuityPolicy().__dict__,
                "continuity_segment_count": len(set(segments)),
                "gaps": gaps.to_dict(orient="records") if not gaps.empty else [],
            }
            if content_changed:
                self._atomic_parquet(path, table)
            if prior_metadata != metadata:
                self._atomic_metadata(metadata_path, metadata)
            if self.catalog is not None:
                provenance = Provenance.create(path, broker, server, symbol, "M1",
                                               first_timestamp.to_pydatetime(), last_timestamp.to_pydatetime(),
                                               len(table), SCHEMA_VERSION)
                self.catalog.upsert_raw_dataset(provenance)
            if pending_path.exists():
                pending_path.unlink()
            outputs.append(path)
        return outputs
