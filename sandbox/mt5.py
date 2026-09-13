from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sandbox.models import SymbolRecord


class MT5Error(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalStatus:
    company: str | None
    server: str | None
    account_currency: str | None
    terminal_name: str | None
    connected: bool


class MT5Adapter:
    """Read-only adapter: exposes terminal metadata, symbols and historical rates only."""

    def __init__(self, terminal_path: Path | None = None, module: Any | None = None):
        self._path = terminal_path
        self._mt5 = module
        self._connected = False

    def __enter__(self) -> "MT5Adapter":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def connect(self) -> TerminalStatus:
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise MT5Error("MetaTrader5 is not installed (Windows and Python 3.12+ required)") from exc
        kwargs = {"path": str(self._path)} if self._path else {}
        if not self._mt5.initialize(**kwargs):
            raise MT5Error(f"MT5 initialization failed: {self._mt5.last_error()}")
        self._connected = True
        return self.status()

    def shutdown(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False

    def _require(self) -> None:
        if not self._connected:
            raise MT5Error("MT5 is not connected")

    def status(self) -> TerminalStatus:
        self._require()
        terminal = self._mt5.terminal_info()
        account = self._mt5.account_info()
        if terminal is None:
            raise MT5Error(f"terminal_info failed: {self._mt5.last_error()}")
        return TerminalStatus(
            getattr(account, "company", None), getattr(account, "server", None),
            getattr(account, "currency", None), getattr(terminal, "name", None),
            bool(getattr(terminal, "connected", True)),
        )

    @staticmethod
    def _currencies(name: str, description: str) -> tuple[str | None, str | None]:
        text = re.sub(r"[^A-Z]", "", name.upper())
        match = re.search(r"(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD)(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD)", text)
        if not match:
            match = re.search(r"\b([A-Z]{3})\s*[/ -]\s*([A-Z]{3})\b", description.upper())
        return match.groups() if match else (None, None)

    def symbols(self) -> list[SymbolRecord]:
        self._require()
        result = []
        for item in self._mt5.symbols_get() or ():
            base, quote = self._currencies(item.name, getattr(item, "description", ""))
            if not base or not quote:
                continue
            result.append(SymbolRecord(
                broker_symbol=item.name, canonical_pair=base + quote,
                base_currency=base, quote_currency=quote, digits=item.digits,
                point=item.point, tick_size=getattr(item, "trade_tick_size", None),
                contract_size=getattr(item, "trade_contract_size", None),
                description=getattr(item, "description", ""), visible=item.visible,
                trade_mode=getattr(item, "trade_mode", None),
            ))
        return sorted(result, key=lambda x: x.broker_symbol)

    def resolve_symbol(self, canonical: str) -> SymbolRecord:
        key = re.sub(r"[^A-Z]", "", canonical.upper())
        matches = [s for s in self.symbols() if s.canonical_pair == key]
        if not matches:
            raise MT5Error(f"No broker symbol found for {canonical}")
        return sorted(matches, key=lambda s: (not s.visible, len(s.broker_symbol), s.broker_symbol))[0]

    def rates(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self._require()
        rows = self._mt5.copy_rates_range(symbol, self._mt5.TIMEFRAME_M1, start, end)
        if rows is None:
            raise MT5Error(f"copy_rates_range failed for {symbol}: {self._mt5.last_error()}")
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
        frame = frame.rename(columns={"time": "timestamp_utc"})
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], unit="s", utc=True)
        return frame[["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

