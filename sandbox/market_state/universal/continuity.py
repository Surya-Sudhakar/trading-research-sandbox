from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class QualityContext:
    """Explicit segment boundaries; feature state is local to each consecutive segment."""
    segment_column: str = "continuity_segment_id"

    def split(self, frame):
        if self.segment_column not in frame:raise ValueError("continuity segment column required")
        times=pd.DatetimeIndex(frame.timestamp_utc)
        if times.tz is None or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("strictly increasing aware timestamps required")
        ids=frame[self.segment_column]
        if ids.isna().any():raise ValueError("missing continuity segment")
        # Repeated IDs must be contiguous; silently merging separated histories is forbidden.
        runs=ids.ne(ids.shift()).cumsum()
        if ids.groupby(runs).first().duplicated().any():raise ValueError("noncontiguous segment ID")
        return (group for _,group in frame.groupby(runs,sort=False))

