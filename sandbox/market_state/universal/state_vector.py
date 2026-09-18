"""Integration audit for the frozen 14-feature market-state V1 vector."""
from dataclasses import dataclass
from typing import Mapping

from .market_state_v1 import (
    FORBIDDEN_STATE_DISCOVERY_FIELDS,
    MARKET_STATE_CONTEXT_V1,
    MARKET_STATE_FEATURE_NAMES_V1,
    MARKET_STATE_FEATURE_SCHEMA_VERSION,
)
from .schema import FeatureRow


@dataclass(frozen=True, slots=True)
class MarketStateVectorV1:
    schema_version: str
    timestamp: object
    symbol: str
    values: tuple[float, ...]
    context: tuple[object, ...]


def vector_ready_v1(row: FeatureRow) -> bool:
    """State readiness is independent from legacy universal feature_ready."""
    return all(getattr(row, name) is not None for name in MARKET_STATE_FEATURE_NAMES_V1)


def project_market_state_v1(row: FeatureRow) -> MarketStateVectorV1 | None:
    if not vector_ready_v1(row):
        return None
    return MarketStateVectorV1(
        schema_version=MARKET_STATE_FEATURE_SCHEMA_VERSION,
        timestamp=row.timestamp,
        symbol=row.symbol,
        values=tuple(float(getattr(row, name)) for name in MARKET_STATE_FEATURE_NAMES_V1),
        context=tuple(getattr(row, name) for name in MARKET_STATE_CONTEXT_V1),
    )


def validate_discovery_columns_v1(columns) -> None:
    """Reject outcome/future fields before any clustering dataset is accepted."""
    lowered = {str(name).strip().lower() for name in columns}
    forbidden = lowered & FORBIDDEN_STATE_DISCOVERY_FIELDS
    if forbidden:
        raise ValueError("forbidden outcome/future fields in state discovery: " + ", ".join(sorted(forbidden)))


def project_mapping_v1(row: Mapping[str, object]) -> tuple[float, ...]:
    """Strict mapping projection used by future artifact builders."""
    validate_discovery_columns_v1(row.keys())
    missing = [name for name in MARKET_STATE_FEATURE_NAMES_V1 if name not in row]
    if missing:
        raise ValueError("missing market-state V1 features: " + ", ".join(missing))
    if any(row[name] is None for name in MARKET_STATE_FEATURE_NAMES_V1):
        raise ValueError("market-state V1 vector is not ready")
    return tuple(float(row[name]) for name in MARKET_STATE_FEATURE_NAMES_V1)
