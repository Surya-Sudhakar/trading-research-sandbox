"""Scalar discovery projection with separate metadata, predictors, and targets."""
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import re

from .research_record import ResearchRecord


@lru_cache(maxsize=None)
def _valid_field_id(field_id):
    return (isinstance(field_id, str)
            and re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", field_id) is not None)


@dataclass(frozen=True, slots=True)
class DiscoveryValue:
    field_id: str
    value: object

    def __post_init__(self):
        if not _valid_field_id(self.field_id):
            raise ValueError("invalid dotted field_id")
        if self.value is not None and not isinstance(self.value, (bool, int, float, str, date, datetime)):
            raise TypeError("value must be an allowed scalar")


@dataclass(frozen=True, slots=True)
class DiscoveryRow:
    record: ResearchRecord
    identity: tuple[DiscoveryValue, ...]
    predictors: tuple[DiscoveryValue, ...]
    targets: tuple[DiscoveryValue, ...]

    def __post_init__(self):
        if not isinstance(self.record, ResearchRecord):
            raise TypeError("record must be ResearchRecord")
        seen = set()
        for field, prefix in (("identity", "meta."), ("predictors", "x."), ("targets", "y.")):
            values = tuple(getattr(self, field))
            for value in values:
                if not isinstance(value, DiscoveryValue):
                    raise TypeError("collections must contain DiscoveryValue")
                if value.field_id in seen:
                    raise ValueError("duplicate field_id")
                seen.add(value.field_id)
                if not value.field_id.startswith(prefix):
                    raise ValueError("field namespace mismatch")
            object.__setattr__(self, field, values)


def values_by_id(values) -> dict[str, object]:
    materialized = tuple(values)
    result = {}
    for value in materialized:
        if not isinstance(value, DiscoveryValue):
            raise TypeError("values must contain DiscoveryValue")
        if value.field_id in result:
            raise ValueError("duplicate field_id")
        result[value.field_id] = value.value
    return result


_TARGET_METRICS = (
    "first_open", "end_close", "highest_high", "lowest_low", "close_change",
    "close_return_fraction", "max_upward_excursion", "max_downward_excursion",
    "bars_to_highest_high", "bars_to_lowest_low",
)


def project_research_record(record: ResearchRecord) -> DiscoveryRow:
    if not isinstance(record, ResearchRecord):
        raise TypeError("record must be ResearchRecord")
    anchor = record.anchor
    identity = tuple(DiscoveryValue("meta." + name, value) for name, value in (
        ("evidence_end_utc", anchor.evidence_end_utc),
        ("geometry_available_at_utc", anchor.geometry_available_at_utc),
        ("source_local_trading_date", anchor.source_local_trading_date),
        ("source_block_start_utc", anchor.source_block_start_utc),
        ("source_block_end_utc", anchor.source_block_end_utc),
        ("outcome_anchor_id", anchor.outcome_anchor.anchor_id),
        ("base_minutes", anchor.base_minutes),
        ("reference_price", anchor.outcome_anchor.reference_price),
        ("level_price", anchor.level_price), ("pattern_index", anchor.pattern_index),
    ))
    predictors = tuple(DiscoveryValue("x." + name, value) for name, value in (
        ("anchor_kind", anchor.kind.value), ("geometry.family_id", anchor.geometry_family_id),
        ("geometry.source_block_label", anchor.source_block_label),
        ("geometry.source_block_index", anchor.source_block_index),
        ("geometry.level_id", anchor.level_id), ("geometry.coordinate", anchor.coordinate),
        ("event.pattern_code", anchor.pattern_code), ("event.checkpoint_bars", anchor.checkpoint_bars),
    ))
    targets = tuple(DiscoveryValue(f"y.h{outcome.horizon_bars}.{metric}", getattr(outcome, metric))
                    for outcome in sorted(record.outcomes, key=lambda o: o.horizon_bars)
                    for metric in _TARGET_METRICS)
    return DiscoveryRow(record, identity, predictors, targets)
