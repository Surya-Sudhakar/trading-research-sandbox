from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone, date

import pytest
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor, ForwardOutcome
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.multitimeframe_context import MultiTimeframeSnapshot
from sandbox.market_state.research_multitimeframe_context import ResearchAnchorMultiTimeframeContext
from sandbox.market_state.research_session_context import ResearchAnchorSessionContext
from sandbox.market_state.research_session_relationship import ResearchAnchorSessionRelationshipContext
from sandbox.market_state.research_record import assemble_research_record
from sandbox.market_state.discovery_projection import DiscoveryValue, DiscoveryRow, project_research_record, values_by_id

TIME = datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)
METRICS = ("first_open", "end_close", "highest_high", "lowest_low", "close_change", "close_return_fraction",
           "max_upward_excursion", "max_downward_excursion", "bars_to_highest_high", "bars_to_lowest_low")


def record(checkpoint=False, horizons=(8, 1, 4)):
    start = TIME - timedelta(hours=8)
    anchor = GeometryResearchAnchor(ResearchAnchorKind.UNTOUCHED_CHECKPOINT if checkpoint else ResearchAnchorKind.TOUCH_TRANSITION,
        "CUSTOM", start.date(), "C1", 0, start, start + timedelta(hours=3), start + timedelta(hours=3),
        "LEVEL", -3.3, 10, TIME, 15, None if checkpoint else 0, None if checkpoint else "ABOVE_TOUCH_BELOW",
        8 if checkpoint else None, OutcomeAnchor("EVIDENCE", TIME, 11))
    mtf = ResearchAnchorMultiTimeframeContext(anchor, MultiTimeframeSnapshot(TIME, OHLCBar(TIME-STEP, 10, 12, 9, 11), ()), (), True)
    sessions = ResearchAnchorSessionContext(anchor, (), (), (), (), True)
    relations = ResearchAnchorSessionRelationshipContext(sessions, (), (), ())
    outcomes = tuple(ForwardOutcome("EVIDENCE", TIME, 11, h, 15, TIME, TIME+h*STEP, TIME, TIME+(h-1)*STEP,
        11, 12, 13, 10, 1, 1/11, 2, 1, TIME, TIME, 0, 0) for h in horizons)
    return assemble_research_record(mtf, relations, outcomes)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_exact_core_fields_and_order(checkpoint):
    source = record(checkpoint)
    row = project_research_record(source)
    assert row.record is source
    a = source.anchor
    expected_meta = {
        "meta.evidence_end_utc": a.evidence_end_utc, "meta.geometry_available_at_utc": a.geometry_available_at_utc,
        "meta.source_local_trading_date": a.source_local_trading_date, "meta.source_block_start_utc": a.source_block_start_utc,
        "meta.source_block_end_utc": a.source_block_end_utc, "meta.outcome_anchor_id": a.outcome_anchor.anchor_id,
        "meta.base_minutes": a.base_minutes, "meta.reference_price": a.outcome_anchor.reference_price,
        "meta.level_price": a.level_price, "meta.pattern_index": a.pattern_index}
    expected_x = {"x.anchor_kind": a.kind.value, "x.geometry.family_id": "CUSTOM", "x.geometry.source_block_label": "C1",
        "x.geometry.source_block_index": 0, "x.geometry.level_id": "LEVEL", "x.geometry.coordinate": -3.3,
        "x.event.pattern_code": None if checkpoint else "ABOVE_TOUCH_BELOW", "x.event.checkpoint_bars": 8 if checkpoint else None}
    assert list(values_by_id(row.identity).items()) == list(expected_meta.items())
    assert list(values_by_id(row.predictors).items()) == list(expected_x.items())
    assert type(row.predictors[0].value) is str
    assert row.identity[0].value is a.evidence_end_utc
    assert row.identity[2].value is a.source_local_trading_date
    assert row == project_research_record(source)


@pytest.mark.parametrize("horizons", [(8, 1, 4), (1, 4, 8, 16, 32)])
def test_target_order_values_and_original_order(horizons):
    source = record(horizons=horizons)
    row = project_research_record(source)
    assert [v.field_id for v in row.targets] == [f"y.h{h}.{m}" for h in sorted(horizons) for m in METRICS]
    target = values_by_id(row.targets)
    for outcome in source.outcomes:
        for metric in METRICS:
            assert target[f"y.h{outcome.horizon_bars}.{metric}"] == getattr(outcome, metric)
    assert tuple(o.horizon_bars for o in source.outcomes) == horizons


def test_none_return_fraction():
    source = record(horizons=(1,))
    source = replace(source, outcomes=(replace(source.outcomes[0], close_return_fraction=None),))
    assert values_by_id(project_research_record(source).targets)["y.h1.close_return_fraction"] is None


def test_mandatory_future_leakage_barrier():
    source = record()
    changed = replace(source, outcomes=tuple(replace(o, end_close=99, highest_high=101, lowest_low=1,
        close_change=88, close_return_fraction=8, max_upward_excursion=90, max_downward_excursion=10) for o in source.outcomes))
    a, b = project_research_record(source), project_research_record(changed)
    assert a.identity == b.identity
    assert a.predictors == b.predictors
    assert a.targets != b.targets
    assert all(v.field_id.startswith("x.") for v in a.predictors)
    assert not any(isinstance(v.value, (date, datetime)) for v in a.predictors)
    assert not any("price" in v.field_id or "utc" in v.field_id for v in a.predictors)


@pytest.mark.parametrize("field_id", ["", "x", "X.field", "x.Bad", "x.1bad", "x..field", "x.field-foo", "x.field\n", 123])
def test_invalid_field_id(field_id):
    with pytest.raises(ValueError):
        DiscoveryValue(field_id, 1)


@pytest.mark.parametrize("value", [[], {}, (), set(), object()])
def test_invalid_scalar(value):
    with pytest.raises(TypeError):
        DiscoveryValue("x.field", value)


@pytest.mark.parametrize("value", [None, True, 1, 1.5, "text", TIME.date(), TIME])
def test_allowed_scalars_preserved(value):
    assert DiscoveryValue("x.field", value).value is value


@pytest.mark.parametrize("field,value", [("identity", DiscoveryValue("x.bad", 1)),
    ("predictors", DiscoveryValue("y.h4.close_change", 1)), ("targets", DiscoveryValue("meta.bad", 1))])
def test_namespace_barrier(field, value):
    row = project_research_record(record())
    with pytest.raises(ValueError):
        replace(row, **{field: (value,)})


def test_duplicates_and_invalid_types():
    source = record()
    value = DiscoveryValue("meta.same", 1)
    with pytest.raises(ValueError):
        DiscoveryRow(source, (value,), (value,), ())
    with pytest.raises(ValueError):
        DiscoveryRow(source, (value, value), (), ())
    with pytest.raises(TypeError):
        DiscoveryRow(object(), (), (), ())
    with pytest.raises(TypeError):
        project_research_record(object())
    with pytest.raises(TypeError):
        DiscoveryRow(source, (object(),), (), ())
    with pytest.raises(TypeError):
        values_by_id([object()])
    with pytest.raises(ValueError):
        values_by_id((value, value))


def test_once_materialization():
    class Once:
        def __init__(self, values):
            self.values, self.calls = values, 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from self.values
    values = (DiscoveryValue("x.b", 2), DiscoveryValue("x.a", 1))
    source = Once(values)
    assert list(values_by_id(source).items()) == [("x.b", 2), ("x.a", 1)]
    assert source.calls == 1
    identity, predictors, targets = Once(()), Once(values), Once(())
    row = DiscoveryRow(record(), identity, predictors, targets)
    assert identity.calls == predictors.calls == targets.calls == 1
    assert row.predictors == values


def test_immutable():
    row = project_research_record(record())
    with pytest.raises(FrozenInstanceError):
        row.targets = ()
    with pytest.raises(FrozenInstanceError):
        row.predictors[0].value = "OTHER"
