"""Immutable assembly of existing causal evidence; no calculations or joins."""
from dataclasses import dataclass
from datetime import timedelta

from .research_anchor import GeometryResearchAnchor
from .research_multitimeframe_context import ResearchAnchorMultiTimeframeContext
from .research_session_relationship import ResearchAnchorSessionRelationshipContext
from .future_outcome import ForwardOutcome
from .discovery_market_context import DiscoveryMarketContext


@dataclass(frozen=True)
class ResearchRecord:
    anchor: GeometryResearchAnchor
    multitimeframe_context: ResearchAnchorMultiTimeframeContext
    session_relationship_context: ResearchAnchorSessionRelationshipContext
    outcomes: tuple[ForwardOutcome, ...]
    market_context: DiscoveryMarketContext | None = None

    def __post_init__(self):
        if not isinstance(self.anchor, GeometryResearchAnchor):
            raise TypeError("anchor must be GeometryResearchAnchor")
        if not isinstance(self.multitimeframe_context, ResearchAnchorMultiTimeframeContext):
            raise TypeError("invalid multitimeframe context")
        if not isinstance(self.session_relationship_context, ResearchAnchorSessionRelationshipContext):
            raise TypeError("invalid session relationship context")
        if (self.multitimeframe_context.anchor != self.anchor
                or self.session_relationship_context.session_context.anchor != self.anchor):
            raise ValueError("context anchor mismatch")
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("at least one outcome is required")
        horizons = set()
        for outcome in outcomes:
            if not isinstance(outcome, ForwardOutcome):
                raise TypeError("outcomes must contain ForwardOutcome")
            for value in (outcome.horizon_bars, outcome.base_minutes):
                if type(value) is not int or value <= 0:
                    raise ValueError("horizon and base_minutes must be positive integers")
            if outcome.horizon_bars in horizons:
                raise ValueError("duplicate horizon")
            horizons.add(outcome.horizon_bars)
            anchor = self.anchor
            if (outcome.anchor_id != anchor.outcome_anchor.anchor_id
                    or outcome.anchor_time_utc != anchor.outcome_anchor.anchor_time_utc
                    or outcome.anchor_time_utc != anchor.evidence_end_utc
                    or outcome.reference_price != anchor.outcome_anchor.reference_price
                    or outcome.base_minutes != anchor.base_minutes):
                raise ValueError("outcome anchor mismatch")
            start = anchor.evidence_end_utc
            step = timedelta(minutes=outcome.base_minutes)
            if (outcome.window_start_utc != start or outcome.first_bar_open_time_utc != start
                    or outcome.window_end_utc != start + outcome.horizon_bars * step
                    or outcome.last_bar_open_time_utc != start + (outcome.horizon_bars - 1) * step):
                raise ValueError("outcome timing mismatch")
        object.__setattr__(self, "outcomes", outcomes)
        if (self.market_context is not None
                and not isinstance(self.market_context, DiscoveryMarketContext)):
            raise TypeError("market_context must be DiscoveryMarketContext")
        if (self.market_context is not None
                and self.market_context.decision_time_utc != self.anchor.evidence_end_utc):
            raise ValueError("market context must match exact evidence time")


def assemble_research_record(multitimeframe_context: ResearchAnchorMultiTimeframeContext,
                             session_relationship_context: ResearchAnchorSessionRelationshipContext,
                             outcomes, market_context: DiscoveryMarketContext | None = None) -> ResearchRecord:
    if not isinstance(multitimeframe_context, ResearchAnchorMultiTimeframeContext):
        raise TypeError("invalid multitimeframe context")
    # ResearchRecord materializes outcomes once and retains all evidence objects.
    return ResearchRecord(multitimeframe_context.anchor, multitimeframe_context,
                          session_relationship_context, outcomes, market_context)
