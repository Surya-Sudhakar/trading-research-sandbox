from dataclasses import dataclass, asdict
from datetime import timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from sandbox.market_state.normalization import timeframe_minutes


@dataclass(frozen=True)
class SessionProfile:
    name: str = "UNKNOWN_PROVIDER_SESSION"
    timezone: str = "UTC"
    # Explicit documented local week-minute boundaries, e.g. Friday 17:00 NY.
    week_close: tuple[int,int,int] | None = None
    week_open: tuple[int,int,int] | None = None
    evidence: str | None = None
    holiday_policy: str = "UNKNOWN"
    dst_behavior: str = "IANA_ZONEINFO"

    def __post_init__(self):
        ZoneInfo(self.timezone)
        if (self.week_close is None) != (self.week_open is None):
            raise ValueError("both weekly boundaries required")
        if self.week_close is not None:
            if not self.evidence:raise ValueError("documented session evidence required")
            for day,hour,minute in (self.week_close,self.week_open):
                if not (0<=day<=6 and 0<=hour<=23 and 0<=minute<=59):
                    raise ValueError("invalid weekly boundary")

    def classify(self, previous, following, minutes):
        if self.week_close is not None:
            local=previous.tz_convert(self.timezone)
            for shift in (-7,0,7):
                monday=(local.normalize()-timedelta(days=local.weekday())).date()+timedelta(days=shift)
                cd,ch,cm=self.week_close;od,oh,om=self.week_open
                close=(pd.Timestamp(monday+timedelta(days=cd))+timedelta(hours=ch,minutes=cm)).tz_localize(self.timezone)
                # Build wall times independently across DST.
                open_date=monday+timedelta(days=od+(7 if od<=cd else 0))
                opened=(pd.Timestamp(open_date)+timedelta(hours=oh,minutes=om)).tz_localize(self.timezone)
                if previous+timedelta(minutes=int(minutes))==close.tz_convert("UTC") and following==opened.tz_convert("UTC"):
                    return "EXPECTED_WEEKEND"
        # This is an association, not an assertion that coverage is valid.
        if previous.weekday()==4 and following.weekday() in (6,0) and following-previous<=timedelta(days=4):
            return "WEEKEND_ASSOCIATED_UNKNOWN"
        return "UNEXPLAINED_GAP"


@dataclass(frozen=True)
class ContinuityPolicy:
    reset_missing_minutes: int = 60
    continue_expected_weekend: bool = True
    reset_invalid_exclusion: bool = True
    version: str = "CONTINUITY_V1"

    def __post_init__(self):
        if self.reset_missing_minutes < 1:raise ValueError("positive reset threshold required")


def inventory(frame, metadata, profile=SessionProfile(), policy=ContinuityPolicy(), excluded_times=()):
    if frame.empty:return pd.DataFrame(), []
    minutes=timeframe_minutes(metadata.timeframe)
    times=pd.DatetimeIndex(frame.timestamp_utc)
    excluded=sorted(pd.Timestamp(t) for t in excluded_times if not pd.isna(t))
    import bisect
    rows=[];segments=[0];segment=0;delta=timedelta(minutes=int(minutes))
    for previous,following in zip(times[:-1],times[1:]):
        missing=int((following-previous)/delta)-1
        if missing>0:
            classification=profile.classify(previous,following,minutes)
            invalid=bisect.bisect_left(excluded,following)>bisect.bisect_right(excluded,previous)
            reset=(policy.reset_invalid_exclusion and invalid) or (
                not (classification=="EXPECTED_WEEKEND" and policy.continue_expected_weekend)
                and (classification=="EXPECTED_WEEKEND" or missing*minutes>=policy.reset_missing_minutes))
            segment+=int(reset)
            rows.append(dict(previous_timestamp=previous.isoformat(),next_timestamp=following.isoformat(),
                             missing_slots=missing,duration_minutes=(following-previous).total_seconds()/60,
                             provider=metadata.provider,symbol=metadata.symbol,timeframe=metadata.timeframe,
                             classification=classification,invalid_bar_exclusion=invalid,
                             reset=bool(reset),next_segment_id=segment))
        segments.append(segment)
    return pd.DataFrame(rows),segments
