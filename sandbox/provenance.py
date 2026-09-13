from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Provenance:
    dataset_id: str
    broker: str
    server: str
    symbol: str
    timeframe: str
    start: str
    end: str
    retrieved_at: str
    row_count: int
    schema_version: str
    checksum: str
    path: str

    @classmethod
    def create(cls, path: Path, broker: str, server: str, symbol: str, timeframe: str, start: datetime, end: datetime, row_count: int, schema_version: str = "1") -> "Provenance":
        return cls(str(uuid4()), broker, server, symbol, timeframe, start.isoformat(), end.isoformat(), datetime.now(timezone.utc).isoformat(), row_count, schema_version, sha256_file(path), str(path))

    def to_dict(self) -> dict:
        return asdict(self)

