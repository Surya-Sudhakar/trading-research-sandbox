from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import math
from pydantic import BaseModel, ConfigDict, model_validator
from sandbox.models import Candle


class DatasetKind(StrEnum):
    RAW = "RAW"
    SANITIZED = "SANITIZED"
    RESAMPLED = "RESAMPLED"
    FEATURE = "FEATURE"


@dataclass(frozen=True)
class ProviderMetadata:
    provider: str
    symbol: str
    provider_symbol: str
    timeframe: str
    price_type: str
    source_dataset_id: str
    timezone: str = "UTC"


class CanonicalMarketBar(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    timestamp_utc: datetime
    symbol: str
    provider: str
    provider_symbol: str
    timeframe: str
    price_type: str
    open: float
    high: float
    low: float
    close: float
    tick_volume: float | None = None
    real_volume: float | None = None
    spread: float | None = None
    source_dataset_id: str
    quality_status: str = "VALID"
    continuity_segment_id: int = 0

    @model_validator(mode="after")
    def validate_values(self):
        Candle(timestamp_utc=self.timestamp_utc,open=self.open,high=self.high,low=self.low,
               close=self.close,tick_volume=0,real_volume=0,spread=0)
        for name in ("tick_volume","real_volume","spread"):
            value=getattr(self,name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid optional market field")
        return self

