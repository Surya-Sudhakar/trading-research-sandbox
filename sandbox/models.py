from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Candle(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp_utc: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    tick_volume: int
    spread: int
    real_volume: int

    @field_validator("timestamp_utc")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def valid_market_values(self) -> "Candle":
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("prices must be positive")
        if not (self.high >= self.open and self.high >= self.close and self.low <= self.open and self.low <= self.close and self.high >= self.low):
            raise ValueError("invalid OHLC relationship")
        if min(self.tick_volume, self.spread, self.real_volume) < 0:
            raise ValueError("volume and spread values cannot be negative")
        return self


class SymbolRecord(BaseModel):
    broker_symbol: str
    canonical_pair: str | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    digits: int
    point: float
    tick_size: float | None = None
    contract_size: float | None = None
    description: str = ""
    visible: bool
    trade_mode: int | None = None

