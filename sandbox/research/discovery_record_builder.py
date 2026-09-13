"""Authorized, segment-local composition of the existing Phase-3 primitives.

Segment-local historical primitives are precomputed once, with their existing
causal availability rules. No second event engine is maintained.
Multiple levels can share an evidence timestamp; their identity breaks ties.
No research record or context is assembled using another segment's bars.
"""
from copy import deepcopy
from itertools import groupby
from dataclasses import dataclass
from datetime import time, timedelta
import re

import pandas as pd

from sandbox.market_state.block_aggregation import OHLCBar, aggregate_completed_blocks
from sandbox.market_state.discovery_dataset import DiscoveryDataset, build_discovery_dataset
from sandbox.market_state.future_outcome import DEFAULT_FORWARD_HORIZONS
from sandbox.market_state.discovery_market_context import (
    build_discovery_market_contexts, with_level_transition,
)
from sandbox.market_state.geometry_lifecycle import (
    BIG_BROTHER_GEOMETRY_FAMILY_ID, build_geometry_instance,
)
from sandbox.market_state.level_interaction_history import build_geometry_interaction_histories
from sandbox.market_state.level_state_sequence import build_level_state_sequence
from sandbox.market_state.price_geometry import BIG_BROTHER_COMMON_LEVELS, ProjectionLevelSpec
from sandbox.market_state.research_anchor import (
    DEFAULT_UNTOUCHED_CHECKPOINTS, build_touch_transition_research_anchor,
    build_untouched_checkpoint_research_anchors, measure_research_anchor_outcomes,
)
from sandbox.market_state.research_multitimeframe_context import join_research_anchors_multitimeframe
from sandbox.market_state.research_record import ResearchRecord, assemble_research_record
from sandbox.market_state.research_session_context import join_research_anchors_sessions
from sandbox.market_state.research_session_relationship import build_research_session_relationships
from sandbox.market_state.session_clock import BlockConfig, NEW_YORK_C1_C8
from sandbox.market_state.touch_transition import (
    build_touch_transition_patterns, classify_level_transition,
)
from sandbox.market_state.universal.continuity import QualityContext
from sandbox.partition.models import AccessContext, AccessOperation, PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.research.errors import ResearchError


_STEP = timedelta(minutes=15)
_M15_BLOCKS = BlockConfig("UTC", time(0), 15, 96, tuple(f"M{i}" for i in range(96)))


@dataclass(frozen=True)
class DiscoveryRecordConfig:
    block_config: BlockConfig = NEW_YORK_C1_C8
    geometry_family_id: str = BIG_BROTHER_GEOMETRY_FAMILY_ID
    level_specs: tuple[ProjectionLevelSpec, ...] = BIG_BROTHER_COMMON_LEVELS
    horizons: tuple[int, ...] = DEFAULT_FORWARD_HORIZONS
    checkpoints: tuple[int, ...] = DEFAULT_UNTOUCHED_CHECKPOINTS
    swing_left_bars: int = 2
    swing_right_bars: int = 2
    swing_count_window: int = 8

    def __post_init__(self):
        if not isinstance(self.block_config, BlockConfig):
            raise TypeError("block_config must be BlockConfig")
        if (not isinstance(self.geometry_family_id, str)
                or re.fullmatch(r"[A-Z0-9_]+", self.geometry_family_id) is None):
            raise ValueError("invalid geometry_family_id")
        levels = tuple(self.level_specs)
        if not levels or any(not isinstance(s, ProjectionLevelSpec) for s in levels):
            raise ValueError("nonempty ProjectionLevelSpec collection required")
        if (len({s.level_id for s in levels}) != len(levels)
                or len({s.coordinate for s in levels}) != len(levels)):
            raise ValueError("duplicate level IDs or coordinates")
        object.__setattr__(self, "level_specs", levels)
        for name in ("horizons", "checkpoints"):
            values = tuple(getattr(self, name))
            if (any(type(n) is not int or n <= 0 for n in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain unique positive integers")
            if name == "horizons" and not values:
                raise ValueError("at least one horizon required")
            object.__setattr__(self, name, values)
        for name in ("swing_left_bars", "swing_right_bars", "swing_count_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DiscoveryRecordPopulation:
    partition_id: str
    partition_fingerprint: str
    source_dataset_id: str
    config: DiscoveryRecordConfig
    records: tuple[ResearchRecord, ...]
    record_segment_ids: tuple[object, ...]
    dataset: DiscoveryDataset


def _segment_bars(frame, timeframe):
    """Validate, never repair gaps or infer missing continuity boundaries."""
    if timeframe not in {"M1", "M15"}:
        raise ValueError("Discovery builder requires an M1 or M15 partition")
    required = {"timestamp_utc", "open", "high", "low", "close", "continuity_segment_id"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("OHLC, timestamp_utc and continuity_segment_id columns required")
    if not frame.columns.is_unique:
        raise ValueError("duplicate dataframe columns")
    source_minutes = 1 if timeframe == "M1" else 15
    source_step = timedelta(minutes=source_minutes)
    for group in QualityContext().split(frame):
        def timestamp(value):
            if isinstance(value, pd.Timestamp) and value.nanosecond == 0:
                return value.to_pydatetime()
            return value
        bars = tuple(OHLCBar(timestamp(row.timestamp_utc), row.open, row.high, row.low, row.close)
                     for row in group.itertuples(index=False))
        if any(right.open_time_utc - left.open_time_utc != source_step
               for left, right in zip(bars, bars[1:])):
            raise ValueError("missing bar inside continuity segment; explicit boundary required")
        completed = aggregate_completed_blocks(
            bars, bars[-1].open_time_utc + source_step, _M15_BLOCKS, source_minutes)
        if timeframe == "M1":
            bars = tuple(OHLCBar(c.block_start_utc, c.open, c.high, c.low, c.close)
                         for c in completed)
        yield group.continuity_segment_id.iloc[0], bars


def _anchor_order(anchor):
    return (anchor.evidence_end_utc, anchor.source_block_start_utc,
            anchor.geometry_family_id, anchor.level_id, anchor.kind.value,
            anchor.pattern_index if anchor.pattern_index is not None else -1,
            anchor.checkpoint_bars if anchor.checkpoint_bars is not None else -1)


def _records_for_segment(bars, config):
    horizon = max(config.horizons)
    eligible_count = len(bars) - horizon
    if eligible_count <= 0:
        return
    prefix = bars[:eligible_count]
    decision = prefix[-1].open_time_utc + _STEP
    anchors = []
    transitions = {}
    for candle in aggregate_completed_blocks(prefix, decision, config.block_config):
        instance = build_geometry_instance(candle, config.geometry_family_id, config.level_specs)
        for history in build_geometry_interaction_histories(instance, prefix, decision):
            sequence = build_level_state_sequence(history)
            for pattern in build_touch_transition_patterns(sequence):
                anchor = build_touch_transition_research_anchor(pattern)
                if anchor is not None:
                    anchors.append(anchor)
                    transitions[_anchor_order(anchor)] = classify_level_transition(pattern)
            anchors.extend(build_untouched_checkpoint_research_anchors(history, config.checkpoints))
    if not anchors:
        return
    anchors.sort(key=_anchor_order)
    market_contexts = build_discovery_market_contexts(
        prefix,
        swing_left_bars=config.swing_left_bars,
        swing_right_bars=config.swing_right_bars,
        swing_count_window=config.swing_count_window,
    )
    context_count = (anchors[-1].evidence_end_utc - bars[0].open_time_utc) // _STEP
    context_bars = prefix[:context_count]
    mtf = join_research_anchors_multitimeframe(anchors, context_bars)
    sessions = join_research_anchors_sessions(anchors, context_bars)
    for _, group in groupby(zip(mtf, sessions), key=lambda pair: pair[0].anchor.evidence_end_utc):
        for mtf_context, session_context in deepcopy(tuple(group)):
            anchor = mtf_context.anchor
            count = (anchor.evidence_end_utc - bars[0].open_time_utc) // _STEP
            market_context = market_contexts.get(anchor.evidence_end_utc)
            if market_context is None:
                raise ValueError("no exact market context at anchor evidence time")
            market_context = with_level_transition(
                market_context, transitions.get(_anchor_order(anchor)))
            yield assemble_research_record(
                mtf_context, build_research_session_relationships(session_context),
                measure_research_anchor_outcomes(anchor, bars[count:count + horizon], config.horizons),
                market_context)


def build_discovery_record_population(
    service: PartitionService, partition_id: str,
    config: DiscoveryRecordConfig = DiscoveryRecordConfig(),
) -> DiscoveryRecordPopulation:
    """Read only via Discovery authorization; return records and their dataset."""
    if not isinstance(config, DiscoveryRecordConfig):
        raise TypeError("config must be DiscoveryRecordConfig")
    manifest = service.get_partition(partition_id)
    if manifest is not None and manifest.role == PartitionRole.DIAGNOSTIC:
        raise ResearchError("Discovery partition required; diagnostic population not supported")
    frame = service.access(partition_id, AccessContext.DISCOVERY_CONTEXT,
                           AccessOperation.FEATURE_ANALYSIS, actor="discovery-record-builder")
    if manifest is None or manifest.role != PartitionRole.DISCOVERY:
        raise ResearchError("Discovery partition required")
    integrity = service.verify_partition(partition_id)
    if not integrity["checksum_valid"] or not integrity["row_count_valid"]:
        raise ResearchError("partition integrity failed")
    records, segment_ids = [], []
    for segment_id, bars in _segment_bars(frame, manifest.timeframe):
        if bars and (bars[0].open_time_utc < manifest.start_timestamp
                     or bars[-1].open_time_utc + _STEP > manifest.end_timestamp):
            raise ValueError("M15 bars must be fully inside partition boundaries")
        for record in _records_for_segment(bars, config):
            records.append(record)
            segment_ids.append(segment_id)
    records = tuple(records)
    return DiscoveryRecordPopulation(
        manifest.partition_id, manifest.partition_fingerprint, manifest.source_dataset_id,
        config, records, tuple(segment_ids), build_discovery_dataset(records))
