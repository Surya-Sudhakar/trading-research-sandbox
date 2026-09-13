from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sandbox import SCHEMA_VERSION, __version__
from sandbox.integrity import audit


COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def normalize_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame[COLUMNS] for frame in chunks if not frame.empty]
    if not available:
        return pd.DataFrame(columns=COLUMNS)
    result = pd.concat(available, ignore_index=True)
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    return result.sort_values("timestamp_utc", kind="mergesort").drop_duplicates("timestamp_utc", keep="first").reset_index(drop=True)


class RawStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def target(self, server: str, symbol: str, month: pd.Timestamp) -> Path:
        return self.data_dir / "raw" / "ic_markets" / safe_segment(server) / safe_segment(symbol) / "M1" / f"{month.year:04d}" / f"{month.month:02d}.parquet"

    def write_months(self, frame: pd.DataFrame, server: str, symbol: str, broker: str, requested_start: datetime, requested_end: datetime) -> list[Path]:
        if frame.empty:
            return []
        clean = normalize_chunks([frame])
        report = audit(clean)
        if any(x.category == "confirmed_data_error" for x in report.issues):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = self.data_dir / "quarantine" / safe_segment(server) / safe_segment(symbol) / f"{stamp}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            clean.to_parquet(path, index=False)
            raise ValueError(f"invalid data quarantined at {path}")
        outputs = []
        for (year, month_number), group in clean.groupby([clean.timestamp_utc.dt.year, clean.timestamp_utc.dt.month]):
            month = pd.Timestamp(year=int(year), month=int(month_number), day=1, tz="UTC")
            path = self.target(server, symbol, month)
            if path.exists():
                raise FileExistsError(f"raw dataset already exists; refusing overwrite: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            table = group.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
            metadata = {
                "broker": broker, "server": server, "broker_symbol": symbol, "timeframe": "M1",
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "requested_start": requested_start.isoformat(),
                "requested_end": requested_end.isoformat(), "actual_start": table.timestamp_utc.min().isoformat(),
                "actual_end": table.timestamp_utc.max().isoformat(), "row_count": len(table),
                "software_version": __version__, "schema_version": SCHEMA_VERSION,
            }
            table.to_parquet(path, index=False)
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            outputs.append(path)
        return outputs
