from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


FEATURE_SCHEMA_VERSION = "MARKET_STATE_V1"
FEATURE_ENGINE_VERSION = "1.0.0"


class SnapshotPhase(StrEnum):
    SETUP_CREATION = "SETUP_CREATION"
    PULLBACK_QUALIFICATION = "PULLBACK_QUALIFICATION"
    ENTRY_DECISION = "ENTRY_DECISION"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class FeatureConfiguration:
    requested_timeframes: tuple[str, ...] = ("M15", "H1")
    horizons: tuple[int, ...] = (4, 8, 12, 24)
    atr_period: int = 14
    volatility_short_window: int = 14
    volatility_long_window: int = 50
    volatility_percentile_history: int = 250
    swing_left_bars: int = 2
    swing_right_bars: int = 2
    swing_count_window: int = 8
    analysis_timezone: str = "America/New_York"
    time_blocks_minutes: tuple[int, ...] = (180,)
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self):
        if not self.requested_timeframes or not self.horizons:
            raise ValueError("timeframes and horizons are required")
        numeric = self.horizons + (self.atr_period, self.volatility_short_window,
            self.volatility_long_window, self.volatility_percentile_history,
            self.swing_left_bars, self.swing_right_bars, self.swing_count_window) + self.time_blocks_minutes
        if any(x < 1 for x in numeric):
            raise ValueError("feature windows must be positive")
        object.__setattr__(self, "requested_timeframes", tuple(self.requested_timeframes))
        object.__setattr__(self, "horizons", tuple(sorted(set(self.horizons))))
        object.__setattr__(self, "time_blocks_minutes", tuple(sorted(set(self.time_blocks_minutes))))

    def scientific_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    feature_name: str
    definition: str
    units: str
    timeframe: str
    lookback_requirements: str
    causal_availability: str
    missing_value_behavior: str
    configuration_dependencies: tuple[str, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MarketStateSnapshot:
    snapshot_id: str
    symbol: str
    decision_timestamp: datetime
    information_cutoff_timestamp: datetime
    base_timeframe: str
    snapshot_phase: SnapshotPhase
    feature_schema_version: str
    feature_engine_code_fingerprint: str
    feature_configuration_fingerprint: str
    values: Mapping[str, Any]
    validity_flags: Mapping[str, bool]
    missing_features: Mapping[str, str]
    data_quality_flags: tuple[str, ...] = ()
    dataset_id: str | None = None
    partition_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "decision_timestamp", _utc(self.decision_timestamp))
        object.__setattr__(self, "information_cutoff_timestamp", _utc(self.information_cutoff_timestamp))
        if self.information_cutoff_timestamp > self.decision_timestamp:
            raise ValueError("information cutoff cannot follow decision time")
        object.__setattr__(self, "snapshot_phase", SnapshotPhase(self.snapshot_phase))
        object.__setattr__(self, "values", _freeze(self.values))
        object.__setattr__(self, "validity_flags", _freeze(self.validity_flags))
        object.__setattr__(self, "missing_features", _freeze(self.missing_features))
        forbidden = {"win", "loss", "pnl", "mfe", "mae"}
        def prohibited(key):
            normalized=key.lower().replace(".","_");parts=set(normalized.split("_"))
            return bool(parts&forbidden) or any(normalized.endswith(x) for x in ("r_result","tp_reached","sl_reached"))
        if any(prohibited(key) for key in self.values):
            raise ValueError("outcome data is prohibited in a pre-entry snapshot")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id, "symbol": self.symbol,
            "decision_timestamp": self.decision_timestamp,
            "information_cutoff_timestamp": self.information_cutoff_timestamp,
            "base_timeframe": self.base_timeframe, "snapshot_phase": self.snapshot_phase,
            "feature_schema_version": self.feature_schema_version,
            "feature_engine_code_fingerprint": self.feature_engine_code_fingerprint,
            "feature_configuration_fingerprint": self.feature_configuration_fingerprint,
            "values": dict(self.values), "validity_flags": dict(self.validity_flags),
            "missing_features": dict(self.missing_features),
            "data_quality_flags": self.data_quality_flags, "dataset_id": self.dataset_id,
            "partition_id": self.partition_id,
        }


@dataclass(frozen=True, slots=True)
class SnapshotLink:
    snapshot_id: str
    snapshot_phase: SnapshotPhase
    setup_id: str | None = None
    signal_id: str | None = None
    trade_id: str | None = None


@dataclass(frozen=True, slots=True)
class MarketStateSnapshotLedger:
    snapshots: tuple[MarketStateSnapshot, ...] = field(default_factory=tuple)
    links: tuple[SnapshotLink, ...] = field(default_factory=tuple)
    ledger_fingerprint: str = ""
