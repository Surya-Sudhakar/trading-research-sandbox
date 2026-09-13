from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class IntegrityIssue:
    category: str
    code: str
    timestamp: str | None
    detail: str


@dataclass(frozen=True)
class IntegrityReport:
    row_count: int
    issues: tuple[IntegrityIssue, ...]

    @property
    def status(self) -> str:
        return "error" if any(x.category == "confirmed_data_error" for x in self.issues) else ("warning" if any(x.category == "suspicious_gap" for x in self.issues) else "clean")

    def to_dict(self) -> dict:
        return {"row_count": self.row_count, "status": self.status, "issues": [asdict(x) for x in self.issues]}


def _weekend_closure(previous: pd.Timestamp, current: pd.Timestamp) -> bool:
    # Typical FX close: Friday 22:00 UTC through Sunday 22:00 UTC; broker DST may shift one hour.
    if previous.weekday() != 4 or current.weekday() != 6:
        return False
    return previous.hour >= 20 and current.hour >= 20 and current.hour <= 23


def audit(frame: pd.DataFrame, expected: timedelta = timedelta(minutes=1), suspicious: timedelta = timedelta(minutes=5)) -> IntegrityReport:
    issues: list[IntegrityIssue] = []
    required = ["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    missing = [c for c in required if c not in frame]
    if missing:
        issues.append(IntegrityIssue("confirmed_data_error", "missing_columns", None, ", ".join(missing)))
        return IntegrityReport(len(frame), tuple(issues))
    if frame[required].isna().any().any():
        issues.append(IntegrityIssue("confirmed_data_error", "null_values", None, "one or more required values are null"))
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if ts.duplicated().any():
        issues.append(IntegrityIssue("confirmed_data_error", "duplicate_timestamp", None, f"{int(ts.duplicated().sum())} duplicate rows"))
    if not ts.is_monotonic_increasing:
        issues.append(IntegrityIssue("confirmed_data_error", "out_of_order", None, "timestamps are not ascending"))
    prices = frame[["open", "high", "low", "close"]]
    invalid = (~np.isfinite(prices)).any(axis=1) | (prices <= 0).any(axis=1) | (frame.high < frame[["open", "close"]].max(axis=1)) | (frame.low > frame[["open", "close"]].min(axis=1)) | (frame.high < frame.low)
    if invalid.any():
        issues.append(IntegrityIssue("confirmed_data_error", "invalid_ohlc", None, f"{int(invalid.sum())} invalid rows"))
    invalid_market_fields=(~np.isfinite(frame[["tick_volume","spread","real_volume"]])).any(axis=1) | (frame[["tick_volume","spread","real_volume"]]<0).any(axis=1)
    if invalid_market_fields.any():issues.append(IntegrityIssue("confirmed_data_error","invalid_volume_or_spread",None,f"{int(invalid_market_fields.sum())} invalid rows"))
    ordered = ts.sort_values().drop_duplicates()
    for previous, current in zip(ordered.iloc[:-1], ordered.iloc[1:]):
        gap = current - previous
        if gap <= expected:
            continue
        if _weekend_closure(previous, current):
            category = "expected_market_closure"
        else:
            category = "suspicious_gap" if gap >= suspicious else "confirmed_data_error"
        issues.append(IntegrityIssue(category, "missing_interval", current.isoformat(), f"gap of {gap}"))
    return IntegrityReport(len(frame), tuple(issues))
