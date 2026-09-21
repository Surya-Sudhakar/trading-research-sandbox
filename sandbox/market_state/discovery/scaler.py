"""Discovery-only scaling for MARKET_STATE_FEATURES_V1.

The scaler is fit strictly on the frozen discovery interval. Validation and
holdout rows may only be transformed by an already-fitted scaler.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.preprocessing import StandardScaler

from sandbox.market_state.universal.market_state_v1 import (
    MARKET_STATE_FEATURE_NAMES_V1,
    MARKET_STATE_FEATURE_SCHEMA_VERSION,
)
from .model import validate_state_matrix


DISCOVERY_START_UTC = datetime(2016, 1, 1, tzinfo=timezone.utc)
DISCOVERY_END_UTC = datetime(2023, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class DiscoveryScalerMetadata:
    schema_version: str
    feature_names: tuple[str, ...]
    fit_start_utc: datetime
    fit_end_exclusive_utc: datetime
    observation_count: int


class DiscoveryStandardScaler:
    """StandardScaler with a hard 2016-2022 fit boundary."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._metadata: DiscoveryScalerMetadata | None = None

    @property
    def metadata(self) -> DiscoveryScalerMetadata:
        if self._metadata is None:
            raise RuntimeError("discovery scaler has not been fitted")
        return self._metadata

    def fit(self, X: np.ndarray, timestamps) -> "DiscoveryStandardScaler":
        values = validate_state_matrix(X)
        stamps = tuple(timestamps)
        if len(stamps) != len(values):
            raise ValueError("timestamps must align one-to-one with observations")
        if not stamps:
            raise ValueError("discovery scaler requires timestamps")
        normalized = tuple(_require_utc(ts) for ts in stamps)
        if any(ts < DISCOVERY_START_UTC or ts >= DISCOVERY_END_UTC for ts in normalized):
            raise ValueError("scaler fit is restricted to discovery interval 2016-2022")
        self._scaler.fit(values)
        self._metadata = DiscoveryScalerMetadata(
            schema_version=MARKET_STATE_FEATURE_SCHEMA_VERSION,
            feature_names=MARKET_STATE_FEATURE_NAMES_V1,
            fit_start_utc=min(normalized),
            fit_end_exclusive_utc=DISCOVERY_END_UTC,
            observation_count=len(values),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._metadata is None:
            raise RuntimeError("discovery scaler has not been fitted")
        return self._scaler.transform(validate_state_matrix(X))

    def save(self, path) -> None:
        from .parameter_artifact import save_payload, scaler_state
        save_payload(path, "DiscoveryStandardScaler", scaler_state(self))

    @classmethod
    def load(cls, path, *, expected_sha256=None, **expected_identity) -> "DiscoveryStandardScaler":
        from .parameter_artifact import load_payload, restore_scaler
        return restore_scaler(load_payload(path, "DiscoveryStandardScaler", expected_sha256=expected_sha256),
                              **expected_identity)


def _require_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC datetimes")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamps must be UTC")
    return value.astimezone(timezone.utc)
