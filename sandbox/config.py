from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    terminal_path: Path | None
    data_dir: Path
    catalog_path: Path
    chunk_days: int
    max_retries: int
    log_level: str

    @classmethod
    def load(cls) -> "Settings":
        _env_file()
        terminal = os.getenv("SANDBOX_MT5_TERMINAL_PATH", "").strip()
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "data"))
        return cls(
            Path(terminal) if terminal else None,
            data_dir,
            Path(os.getenv("SANDBOX_CATALOG_PATH", str(data_dir / "catalog.sqlite3"))),
            max(1, int(os.getenv("SANDBOX_DEFAULT_CHUNK_DAYS", "7"))),
            max(1, int(os.getenv("SANDBOX_MAX_RETRIES", "3"))),
            os.getenv("SANDBOX_LOG_LEVEL", "INFO").upper(),
        )

