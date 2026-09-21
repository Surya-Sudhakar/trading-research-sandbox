from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(StrEnum):
    MARKET_NEXT_OPEN = "MARKET_NEXT_OPEN"
    MARKET_CLOSE = "MARKET_CLOSE"
    LIMIT = "LIMIT"
    STOP = "STOP"


class PositionStatus(StrEnum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


class SameBarPolicy(StrEnum):
    CONSERVATIVE = "CONSERVATIVE"


class SpreadModel(StrEnum):
    ZERO_SPREAD = "ZERO_SPREAD"
    FIXED_SPREAD = "FIXED_SPREAD"
    HISTORICAL_CANDLE_SPREAD = "HISTORICAL_CANDLE_SPREAD"


class CommissionModel(StrEnum):
    ZERO_COMMISSION = "ZERO_COMMISSION"
    FIXED_PER_TRADE = "FIXED_PER_TRADE"
    FIXED_PER_LOT_PER_SIDE = "FIXED_PER_LOT_PER_SIDE"


class SlippageModel(StrEnum):
    ZERO_SLIPPAGE = "ZERO_SLIPPAGE"
    FIXED_PRICE_SLIPPAGE = "FIXED_PRICE_SLIPPAGE"


class GapPolicy(StrEnum):
    CONSERVATIVE_OPEN = "CONSERVATIVE_OPEN"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class TradeIntent(FrozenModel):
    trade_intent_id: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    direction: Direction
    signal_timestamp: datetime
    requested_entry_type: EntryType
    requested_entry_price: float | None = None
    expires_at: datetime | None = None
    stop_loss: float
    take_profit: float
    quantity: float = Field(default=1.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("signal_timestamp", "expires_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


class ExecutionConfig(FrozenModel):
    same_bar_policy: SameBarPolicy = SameBarPolicy.CONSERVATIVE
    spread_model: SpreadModel = SpreadModel.ZERO_SPREAD
    fixed_spread_price: float = Field(default=0.0, ge=0)
    symbol_point: float = Field(default=0.00001, gt=0)
    commission_model: CommissionModel = CommissionModel.ZERO_COMMISSION
    fixed_commission: float = Field(default=0.0, ge=0)
    commission_per_lot_per_side: float = Field(default=0.0, ge=0)
    slippage_model: SlippageModel = SlippageModel.ZERO_SLIPPAGE
    fixed_slippage_price: float = Field(default=0.0, ge=0)
    gap_policy: GapPolicy = GapPolicy.CONSERVATIVE_OPEN
    candle_interval_seconds: int = Field(default=60, gt=0)
    entry_policy_version: str = "1"


class Position(FrozenModel):
    position_id: str
    trade_intent_id: str
    symbol: str
    direction: Direction
    signal_timestamp: datetime
    entry_timestamp: datetime | None
    entry_price: float | None
    stop_loss: float
    take_profit: float
    quantity: float
    initial_risk_price_distance: float | None
    status: PositionStatus


class LedgerRecord(FrozenModel):
    trade_id: str
    trade_intent_id: str
    strategy_id: str
    experiment_id: str
    dataset_id: str
    symbol: str
    direction: Direction
    signal_timestamp: datetime
    entry_timestamp: datetime | None = None
    entry_price: float | None = None
    stop_loss: float
    take_profit: float
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_reason: str
    rejection_reason: str | None = None
    gross_pnl: float = 0.0
    spread_cost: float = 0.0
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    gross_r: float | None = None
    net_r: float | None = None
    bars_held: int = 0
    market_gap_count: int = 0
    quantity: float = 1.0
    initial_risk_price_distance: float | None = None
    status: PositionStatus
    intent_metadata_json: str = "{}"
