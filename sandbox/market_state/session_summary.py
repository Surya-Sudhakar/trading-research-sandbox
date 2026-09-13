"""Independent completed recurring sessions; no trading signals."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
import re
from zoneinfo import ZoneInfo

from .block_aggregation import OHLCBar

UTC = timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")


@dataclass(frozen=True)
class SessionConfig:
    session_id: str
    timezone: str
    local_start: time
    local_end: time

    def __post_init__(self):
        if not isinstance(self.session_id, str) or re.fullmatch(r"[A-Z0-9_]+", self.session_id) is None:
            raise ValueError("invalid session_id")
        ZoneInfo(self.timezone)
        for boundary in (self.local_start, self.local_end):
            if not isinstance(boundary, time) or boundary.tzinfo is not None:
                raise ValueError("session boundaries must be naive local times")
        if self.local_start == self.local_end:
            raise ValueError("session start and end must differ")


TOKYO_SESSION = SessionConfig("TOKYO", "Asia/Tokyo", time(9), time(18))
LONDON_SESSION = SessionConfig("LONDON", "Europe/London", time(8), time(17))
NEW_YORK_SESSION = SessionConfig("NEW_YORK", "America/New_York", time(8), time(17))
DEFAULT_SESSIONS = (TOKYO_SESSION, LONDON_SESSION, NEW_YORK_SESSION)


@dataclass(frozen=True)
class SessionWindow:
    session_id: str
    local_session_date: date
    local_start: datetime
    local_end: datetime
    start_utc: datetime
    end_utc: datetime


def session_window_for_date(config: SessionConfig, local_session_date: date) -> SessionWindow:
    """Resolve local boundaries; time.fold selects ambiguous occurrences.

    Nonexistent DST boundary times are rejected rather than silently shifted.
    Sessions crossing transitions with valid boundaries use actual UTC elapsed time.
    """
    if not isinstance(local_session_date, date) or isinstance(local_session_date, datetime):
        raise ValueError("local_session_date must be a date")
    zone = ZoneInfo(config.timezone)
    end_date = local_session_date + timedelta(days=config.local_end < config.local_start)
    start = datetime.combine(local_session_date, config.local_start, zone)
    end = datetime.combine(end_date, config.local_end, zone)
    for boundary in (start, end):
        roundtrip = boundary.astimezone(UTC).astimezone(zone)
        if roundtrip.replace(tzinfo=None) != boundary.replace(tzinfo=None):
            raise ValueError("nonexistent local session boundary")
    start_utc, end_utc = start.astimezone(UTC), end.astimezone(UTC)
    if end_utc <= start_utc:
        raise ValueError("session must have positive elapsed UTC duration")
    return SessionWindow(config.session_id, local_session_date, start, end, start_utc, end_utc)


class SessionDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    FLAT = "FLAT"


@dataclass(frozen=True)
class CompletedSessionSummary:
    session_id: str
    local_session_date: date
    start_utc: datetime
    end_utc: datetime
    local_start: datetime
    local_end: datetime
    open: float
    high: float
    low: float
    close: float
    range: float
    direction: SessionDirection
    base_bar_count: int
    expected_base_bar_count: int


def aggregate_completed_sessions(bars, decision_time_utc, sessions=DEFAULT_SESSIONS,
                                 base_minutes=15) -> tuple[CompletedSessionSummary, ...]:
    """Aggregate each session independently on exact UTC grids, sorted by start.

    Boundaries use [start, end). Input alignment uses the UTC epoch grid.
    Incomplete windows are omitted, including those with missing boundary bars.
    """
    _utc(decision_time_utc)
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    sessions = tuple(sessions)
    if any(not isinstance(s, SessionConfig) for s in sessions):
        raise ValueError("sessions must contain SessionConfig")
    if len({s.session_id for s in sessions}) != len(sessions):
        raise ValueError("duplicate session_id")
    step = timedelta(minutes=base_minutes)
    available = {}
    previous = None
    for bar in bars:
        if not isinstance(bar, OHLCBar):
            raise ValueError("bars must contain OHLCBar")
        stamp = bar.open_time_utc
        _utc(stamp)
        if previous is not None and stamp <= previous:
            raise ValueError("timestamps must be strictly increasing")
        previous = stamp
        if (stamp - _EPOCH) % step:
            raise ValueError("timestamp is not aligned to base timeframe")
        if stamp + step <= decision_time_utc:
            available[stamp] = bar
    if not available:
        return ()
    summaries = []
    for config in sessions:
        zone = ZoneInfo(config.timezone)
        # Candidate dates derive only from completed supplied bars, not future data.
        dates = {stamp.astimezone(zone).date() for stamp in available}
        if config.local_end < config.local_start:
            dates |= {d - timedelta(days=1) for d in tuple(dates)}
        for day in sorted(dates):
            window = session_window_for_date(config, day)
            if window.end_utc > decision_time_utc:
                continue
            elapsed = window.end_utc - window.start_utc
            if elapsed % step:
                raise ValueError("elapsed session duration must be divisible by base_minutes")
            expected = elapsed // step
            members = [available.get(window.start_utc + i * step) for i in range(expected)]
            if any(bar is None for bar in members):
                continue
            opening, closing = members[0].open, members[-1].close
            high, low = max(b.high for b in members), min(b.low for b in members)
            direction = SessionDirection.BULLISH if closing > opening else (
                SessionDirection.BEARISH if closing < opening else SessionDirection.FLAT)
            summaries.append(CompletedSessionSummary(
                config.session_id, day, window.start_utc, window.end_utc,
                window.local_start, window.local_end, opening, high, low, closing,
                high - low, direction, len(members), expected))
    return tuple(sorted(summaries, key=lambda s: (s.start_utc, s.end_utc, s.session_id)))
