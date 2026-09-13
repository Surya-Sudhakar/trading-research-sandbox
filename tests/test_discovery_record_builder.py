from collections import Counter
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from sandbox.market_state.discovery_dataset import build_discovery_dataset
from sandbox.market_state.research_anchor import ResearchAnchorKind
from sandbox.partition.models import PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.provenance import Provenance
from sandbox.research.errors import ResearchError
from sandbox.research.registry import ResearchRegistry
from sandbox.research import discovery_record_builder as builder


START = pd.Timestamp("2024-01-15T05:00:00Z")  # NY C1, winter.
STEP = timedelta(minutes=15)


@pytest.fixture
def tmp_path():
    # Match the repository's runtime fixture convention (Windows ACL-safe).
    path = Path("tests/runtime/discovery_record_builder") / uuid4().hex
    path.mkdir(parents=True)
    return path.resolve()


def market_frame(count=48, start=START, segment=0, touches=False):
    rows = []
    for i in range(count):
        if i < 12:
            opening, closing, low, high = 100., 101., 99.9, 101.1
        else:
            price = (102., 103.3, 104., 104.)[(i - 12) % 4] if touches else 102.
            opening = closing = price
            low, high = price - (0.5 if price == 103.3 else 0.1), price + (0.5 if price == 103.3 else 0.1)
        rows.append(dict(timestamp_utc=start + i * STEP, open=opening, high=high,
                         low=low, close=closing, tick_volume=1, spread=0,
                         real_volume=0, continuity_segment_id=segment))
    return pd.DataFrame(rows)


def partition(tmp_path, frame, role=PartitionRole.DISCOVERY, timeframe="M15"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ResearchRegistry(tmp_path / "catalog.sqlite", project_root=Path.cwd(),
                                results_root=tmp_path / "results")
    source = tmp_path / "source.parquet"
    frame.to_parquet(source, index=False)
    first, last = frame.timestamp_utc.iloc[0], frame.timestamp_utc.iloc[-1]
    provenance = Provenance.create(source, "Synthetic", "BuilderTest", "X", timeframe,
                                   first, last, len(frame))
    registry.catalog.add_dataset(provenance)
    service = PartitionService(registry, tmp_path / "partitions", tmp_path / "vault")
    manifest = service.create_partition(
        provenance.dataset_id, "BUILDER_TEST", role, "X", timeframe,
        first, last + timedelta(minutes=1 if timeframe == "M1" else 15))
    return service, manifest


def run(tmp_path, frame=None, config=builder.DiscoveryRecordConfig()):
    service, manifest = partition(tmp_path, market_frame() if frame is None else frame)
    return builder.build_discovery_record_population(service, manifest.partition_id, config)


def test_authorized_dataframe_vertical_slice_repeatable_dataset_and_provenance(tmp_path, monkeypatch):
    service, manifest = partition(tmp_path, market_frame(64, touches=True))
    reads = []
    original_read = pd.read_parquet
    def partition_only(path, *args, **kwargs):
        reads.append(Path(path))
        assert Path(path).name == manifest.partition_id + ".parquet"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", partition_only)
    first = builder.build_discovery_record_population(service, manifest.partition_id)
    second = builder.build_discovery_record_population(service, manifest.partition_id)
    assert first == second
    assert first.records and first.dataset == build_discovery_dataset(first.records)
    assert first.partition_id == manifest.partition_id
    assert first.partition_fingerprint == manifest.partition_fingerprint
    assert first.source_dataset_id == manifest.source_dataset_id
    counts = Counter(r.anchor.kind for r in first.records)
    assert counts[ResearchAnchorKind.TOUCH_TRANSITION] > 0
    assert counts[ResearchAnchorKind.UNTOUCHED_CHECKPOINT] > 0
    assert {r.anchor.geometry_family_id for r in first.records} == {"BIG_BROTHER_COMMON"}
    keys = [builder._anchor_order(r.anchor) for r in first.records]
    # Evidence timestamps may tie across levels, but complete ordering is strict.
    assert all(a < b for a, b in zip(keys, keys[1:]))
    history = service.access_history(manifest.partition_id)
    assert len(history) == 2
    assert all(h["decision"] == "ALLOWED" and h["access_context"] == "DISCOVERY_CONTEXT"
               and h["requested_operation"] == "FEATURE_ANALYSIS" for h in history)
    assert service.verify_access_ledger()
    assert reads
    print("vertical slice:", len(first.records), {k.value: v for k, v in counts.items()})


@pytest.mark.parametrize("role", [PartitionRole.VALIDATION, PartitionRole.FINAL_TEST])
def test_protected_partitions_denied_before_parquet_read(tmp_path, monkeypatch, role):
    service, manifest = partition(tmp_path, market_frame(), role)
    def forbidden(*args, **kwargs):
        pytest.fail("protected market data must not be read")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ResearchError, match="ACCESS_DENIED"):
        builder.build_discovery_record_population(service, manifest.partition_id)
    assert service.access_history(manifest.partition_id)[-1]["decision"] == "DENIED"


@pytest.mark.parametrize("size, expected", [(47, 0), (48, 3)])
def test_full_32_bar_future_is_required(tmp_path, size, expected):
    result = run(tmp_path, market_frame(size))
    assert len(result.records) == expected
    for record in result.records:
        assert record.anchor.kind == ResearchAnchorKind.UNTOUCHED_CHECKPOINT
        assert record.anchor.checkpoint_bars == 4
        assert record.anchor.evidence_end_utc == START + 16 * STEP
        assert tuple(o.horizon_bars for o in record.outcomes) == (1, 4, 8, 16, 32)
        assert record.outcomes[-1].window_end_utc == START + 48 * STEP


def test_adjacent_segments_do_not_complete_future_or_geometry(tmp_path):
    first = market_frame(47)
    second = market_frame(48, START + 47 * STEP, 1)
    joined = run(tmp_path / "joined", pd.concat([first, second], ignore_index=True))
    separate = run(tmp_path / "separate", second)
    assert joined.records == separate.records
    assert all(s == 1 for s in joined.record_segment_ids)
    assert all(r.anchor.source_block_start_utc >= second.timestamp_utc.iloc[0]
               for r in joined.records)


@pytest.mark.parametrize("gap", [timedelta(0), timedelta(days=2)])
def test_segment_reset_equals_independent_runs_including_context(tmp_path, gap):
    first = market_frame(64, touches=True)
    second = market_frame(64, START + 64 * STEP + gap, 1, touches=True)
    together = run(tmp_path / "together", pd.concat([first, second], ignore_index=True))
    left, right = run(tmp_path / "left", first), run(tmp_path / "right", second)
    assert left.records and right.records
    assert together.records == left.records + right.records
    assert together.record_segment_ids == (0,) * len(left.records) + (1,) * len(right.records)
    for record, segment in zip(together.records, together.record_segment_ids):
        frame = first if segment == 0 else second
        start, end = frame.timestamp_utc.iloc[0], frame.timestamp_utc.iloc[-1] + STEP
        assert start <= record.anchor.source_block_start_utc
        assert record.outcomes[-1].window_end_utc <= end
        for slot in record.multitimeframe_context.snapshot.frames:
            if slot.candle is not None:
                assert start <= slot.candle.block_start_utc < slot.candle.block_end_utc <= record.anchor.evidence_end_utc
        for slot in record.session_relationship_context.session_context.sessions:
            if slot.summary is not None:
                assert start <= slot.summary.start_utc < slot.summary.end_utc <= record.anchor.evidence_end_utc


def test_unfinished_touch_cannot_be_confirmed_in_next_segment(tmp_path):
    first = market_frame(16)
    first.loc[14:15, ["open", "high", "low", "close"]] = [103.3, 103.8, 102.8, 103.3]
    second = market_frame(64, START + 16 * STEP, 1)
    second.loc[:3, ["open", "high", "low", "close"]] = [104., 104.1, 103.9, 104.]
    isolated = run(tmp_path / "isolated", pd.concat([first, second], ignore_index=True))
    merged = pd.concat([first, second], ignore_index=True)
    merged["continuity_segment_id"] = 0
    continuous = run(tmp_path / "continuous", merged)
    assert any(r.anchor.source_block_start_utc == START
               and r.anchor.kind == ResearchAnchorKind.TOUCH_TRANSITION
               for r in continuous.records)
    assert not any(r.anchor.source_block_start_utc == START for r in isolated.records)


def test_outcomes_only_post_anchor_and_future_changes_do_not_change_predictors(tmp_path):
    original = market_frame()
    changed = original.copy()
    changed.loc[16:, ["open", "high", "low", "close"]] = [200., 201., 199., 200.]
    before, after = run(tmp_path / "before", original), run(tmp_path / "after", changed)
    assert before.records and len(before.records) == len(after.records)
    for left, right in zip(before.dataset.rows, after.dataset.rows):
        assert left.identity == right.identity and left.predictors == right.predictors
        assert left.targets != right.targets
        assert left.record.anchor == right.record.anchor
        for outcome in right.record.outcomes:
            assert outcome.first_bar_open_time_utc == right.record.anchor.evidence_end_utc
            assert outcome.highest_high == 201. and outcome.lowest_low == 199.


def test_causal_consumers_receive_prefix_only(tmp_path, monkeypatch):
    calls = Counter()
    def wrap(name, index):
        original = getattr(builder, name)
        def checked(*args, **kwargs):
            bars = args[index]
            if name == "build_geometry_interaction_histories":
                evidence = args[2]
            elif name == "measure_research_anchor_outcomes":
                evidence = args[0].evidence_end_utc
                assert bars[0].open_time_utc == evidence
                calls[name] += 1
                return original(*args, **kwargs)
            else:
                evidence = max(a.evidence_end_utc for a in args[0])
            assert all(b.open_time_utc + STEP <= evidence for b in bars)
            calls[name] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(builder, name, checked)
    for name in ("build_geometry_interaction_histories", "join_research_anchors_multitimeframe",
                 "join_research_anchors_sessions", "measure_research_anchor_outcomes"):
        wrap(name, 1)
    assert run(tmp_path).records
    assert len(calls) == 4 and all(calls.values())


@pytest.mark.parametrize("case", ["duplicate", "reversed", "naive", "nat", "offgrid", "gap", "missing_segment", "null_segment", "reused_segment", "bad_ohlc"])
def test_bad_bars_rejected_without_repair(case):
    frame = market_frame()
    if case == "duplicate": frame.loc[1, "timestamp_utc"] = frame.loc[0, "timestamp_utc"]
    elif case == "reversed": frame = frame.iloc[::-1]
    elif case == "naive": frame["timestamp_utc"] = frame.timestamp_utc.dt.tz_localize(None)
    elif case == "nat": frame.loc[1, "timestamp_utc"] = pd.NaT
    elif case == "offgrid": frame["timestamp_utc"] += timedelta(seconds=1)
    elif case == "gap": frame = frame.drop(index=2)
    elif case == "missing_segment": frame = frame.drop(columns="continuity_segment_id")
    elif case == "null_segment": frame["continuity_segment_id"] = None
    elif case == "reused_segment": frame.loc[10:20, "continuity_segment_id"] = 1
    elif case == "bad_ohlc": frame.loc[0, "high"] = 1.
    with pytest.raises((ValueError, TypeError)):
        tuple(builder._segment_bars(frame, "M15"))


def test_m1_complete_segment_local_buckets_reuse_aggregation(tmp_path):
    m15 = market_frame()
    rows = []
    for row in m15.to_dict("records"):
        for minute in range(15):
            rows.append({**row, "timestamp_utc": row["timestamp_utc"] + timedelta(minutes=minute)})
    service, manifest = partition(tmp_path / "m1", pd.DataFrame(rows), timeframe="M1")
    result = builder.build_discovery_record_population(service, manifest.partition_id)
    assert result.records == run(tmp_path / "m15", m15).records
    split = pd.DataFrame(rows)
    split.loc[7:, "continuity_segment_id"] = 1
    segments = tuple(builder._segment_bars(split, "M1"))
    assert not segments[0][1]
    assert segments[1][1][0].open_time_utc == START + STEP


@pytest.mark.parametrize("kwargs", [dict(horizons=()), dict(horizons=(0,)), dict(horizons=(1, 1)),
    dict(checkpoints=(True,)), dict(level_specs=()), dict(geometry_family_id="bad id")])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): builder.DiscoveryRecordConfig(**kwargs)


def test_configuration_and_result_are_frozen(tmp_path):
    config = builder.DiscoveryRecordConfig(horizons=[1, 4, 8, 16, 32], checkpoints=[4])
    result = run(tmp_path, config=config)
    assert isinstance(config.horizons, tuple) and isinstance(result.records, tuple)
    with pytest.raises(FrozenInstanceError): config.horizons = (1,)
    with pytest.raises(FrozenInstanceError): result.records = ()
    assert run(tmp_path / "no_checkpoint", config=replace(config, checkpoints=())).records == ()
