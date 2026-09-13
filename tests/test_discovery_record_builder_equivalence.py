from dataclasses import asdict
from datetime import timedelta
from collections import Counter

import pandas as pd
import pytest

from discovery_record_reference import reference_records_for_segment
from test_discovery_record_builder import market_frame, START, STEP
from sandbox.research import discovery_record_builder as builder
from sandbox.research.discovery_runner_artifacts import identity
from sandbox.market_state.discovery_dataset import build_discovery_dataset


def fixtures(case):
    if case == "short":
        return market_frame(47)
    if case == "complete":
        return market_frame(48)
    if case == "transitions":
        return market_frame(96, touches=True)
    if case == "completed_day":
        return market_frame(160, touches=True)
    if case == "dst_spring":
        return market_frame(120, pd.Timestamp("2024-03-10T05:00:00Z"), touches=True)
    if case == "dst_fall":
        return market_frame(120, pd.Timestamp("2024-11-03T04:00:00Z"), touches=True)
    first = market_frame(64, touches=True)
    start = START + 64 * STEP + (timedelta(days=2) if case == "gap" else timedelta(0))
    return pd.concat([first, market_frame(64, start, 1, touches=True)], ignore_index=True)


def population(frame, function):
    return tuple(record for _, bars in builder._segment_bars(frame, "M15")
                 for record in function(bars, builder.DiscoveryRecordConfig()))


@pytest.mark.parametrize("case", ["short", "complete", "transitions", "completed_day", "dst_spring", "dst_fall", "gap", "adjacent"])
def test_every_record_field_dataset_and_fingerprint_equal(case):
    frame = fixtures(case)
    # Retain original pandas scalar timestamps in the oracle as well.
    old = tuple(record for group in builder.QualityContext().split(frame)
                for record in reference_records_for_segment(tuple(
                    builder.OHLCBar(row.timestamp_utc, row.open, row.high, row.low, row.close)
                    for row in group.itertuples(index=False)), builder.DiscoveryRecordConfig()))
    new = population(frame, builder._records_for_segment)
    assert new == old
    assert [asdict(r) for r in new] == [asdict(r) for r in old]
    assert identity(new) == identity(old)
    old_dataset, new_dataset = build_discovery_dataset(old), build_discovery_dataset(new)
    assert new_dataset == old_dataset
    assert identity(new_dataset) == identity(old_dataset)
    if case == "completed_day":
        assert any(slot.frame_id == "NY_D1" and slot.candle is not None
                   for record in new for slot in record.multitimeframe_context.snapshot.frames)


def test_future_changes_cannot_change_existing_anchor_or_context():
    frame = market_frame(96, touches=True)
    changed = frame.copy()
    changed.loc[40:, ["open", "high", "low", "close"]] = [200., 201., 199., 200.]
    before = population(frame, builder._records_for_segment)
    after = population(changed, builder._records_for_segment)
    select = lambda records: tuple((r.anchor, r.multitimeframe_context, r.session_relationship_context)
                                   for r in records if r.anchor.evidence_end_utc <= START + 40 * STEP)
    assert select(before) and select(before) == select(after)


def test_histories_once_per_geometry_and_context_once_per_segment(monkeypatch):
    counts = Counter()
    for name in ("build_geometry_instance", "build_geometry_interaction_histories",
                 "build_level_state_sequence", "join_research_anchors_multitimeframe",
                 "join_research_anchors_sessions"):
        original = getattr(builder, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(builder, name, counted)
    assert population(market_frame(96, touches=True), builder._records_for_segment)
    assert counts["build_geometry_instance"] == 5
    assert counts["build_geometry_interaction_histories"] == 5
    assert counts["build_level_state_sequence"] == 15
    assert counts["join_research_anchors_multitimeframe"] == 1
    assert counts["join_research_anchors_sessions"] == 1


def test_nanosecond_timestamps_are_not_silently_rounded():
    frame = market_frame(48)
    frame["timestamp_utc"] += pd.Timedelta(1, unit="ns")
    with pytest.raises(ValueError, match="aligned"):
        tuple(builder._segment_bars(frame, "M15"))
