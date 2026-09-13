"""Exact causal joins to a single precomputed historical context stream."""
from dataclasses import dataclass
from datetime import timedelta

from .research_anchor import GeometryResearchAnchor
from .multitimeframe_context import ContextFrame, DEFAULT_CONTEXT_FRAMES, MultiTimeframeSnapshot
from .historical_multitimeframe import iter_historical_multitimeframe_snapshots
from .block_aggregation import OHLCBar


@dataclass(frozen=True)
class ResearchAnchorMultiTimeframeContext:
    anchor: GeometryResearchAnchor
    snapshot: MultiTimeframeSnapshot
    missing_frame_ids: tuple[str, ...]
    all_frames_available: bool

    def __post_init__(self):
        if not isinstance(self.anchor, GeometryResearchAnchor):
            raise TypeError("anchor must be GeometryResearchAnchor")
        if not isinstance(self.snapshot, MultiTimeframeSnapshot):
            raise TypeError("snapshot must be MultiTimeframeSnapshot")
        if self.snapshot.decision_time_utc != self.anchor.evidence_end_utc:
            raise ValueError("snapshot must match exact evidence time")
        bar = self.snapshot.base_bar
        if not isinstance(bar, OHLCBar):
            raise ValueError("snapshot must contain the evidence base bar")
        if bar.open_time_utc + timedelta(minutes=self.anchor.base_minutes) != self.anchor.evidence_end_utc:
            raise ValueError("base bar must end at evidence time")
        if bar.close != self.anchor.outcome_anchor.reference_price:
            raise ValueError("base bar close disagrees with anchor reference price")
        missing = tuple(f.frame_id for f in self.snapshot.frames if f.candle is None)
        if tuple(self.missing_frame_ids) != missing or self.all_frames_available != (not missing):
            raise ValueError("inconsistent missing frame summary")
        object.__setattr__(self, "missing_frame_ids", missing)


def join_research_anchors_multitimeframe(anchors, bars, frames=DEFAULT_CONTEXT_FRAMES,
                                        base_minutes: int = 15) -> tuple[ResearchAnchorMultiTimeframeContext, ...]:
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    anchors = tuple(anchors)
    for anchor in anchors:
        if not isinstance(anchor, GeometryResearchAnchor):
            raise TypeError("anchors must contain GeometryResearchAnchor")
        if anchor.base_minutes != base_minutes:
            raise ValueError("anchor base_minutes mismatch")
    frames = tuple(frames)
    if any(not isinstance(frame, ContextFrame) for frame in frames):
        raise TypeError("frames must contain ContextFrame")
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("duplicate frame IDs")
    if not anchors:
        return ()
    # The existing iterator materializes and validates the supplied bars once.
    lookup = {s.decision_time_utc: s for s in
              iter_historical_multitimeframe_snapshots(bars, frames, base_minutes)}
    result = []
    for anchor in anchors:
        snapshot = lookup.get(anchor.evidence_end_utc)
        if snapshot is None:
            raise ValueError("no exact snapshot at anchor evidence time")
        missing = tuple(frame.frame_id for frame in snapshot.frames if frame.candle is None)
        result.append(ResearchAnchorMultiTimeframeContext(anchor, snapshot, missing, not missing))
    return tuple(result)


def join_research_anchor_multitimeframe(anchor: GeometryResearchAnchor, bars,
                                       frames=DEFAULT_CONTEXT_FRAMES) -> ResearchAnchorMultiTimeframeContext:
    if not isinstance(anchor, GeometryResearchAnchor):
        raise TypeError("anchor must be GeometryResearchAnchor")
    return join_research_anchors_multitimeframe((anchor,), bars, frames, anchor.base_minutes)[0]
