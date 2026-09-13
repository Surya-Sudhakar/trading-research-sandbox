from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor, ForwardOutcome
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.multitimeframe_context import MultiTimeframeSnapshot
from sandbox.market_state.research_multitimeframe_context import ResearchAnchorMultiTimeframeContext
from sandbox.market_state.research_session_context import ResearchAnchorSessionContext
from sandbox.market_state.research_session_relationship import ResearchAnchorSessionRelationshipContext
from sandbox.market_state.research_record import assemble_research_record
from sandbox.market_state.discovery_projection import DiscoveryValue, DiscoveryRow
from sandbox.market_state.discovery_context_projection import project_research_record_with_context
from sandbox.market_state import discovery_dataset as module

TIME = datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def record(label="A", horizons=(1,4,8)):
    start=TIME-timedelta(hours=8)
    anchor=GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION,"CUSTOM",start.date(),"C1",0,start,
        start+timedelta(hours=3),start+timedelta(hours=3),label,0,10,TIME,15,0,"ABOVE_TOUCH_BELOW",None,
        OutcomeAnchor("EVIDENCE",TIME,11))
    mtf=ResearchAnchorMultiTimeframeContext(anchor,MultiTimeframeSnapshot(TIME,OHLCBar(TIME-STEP,10,12,9,11),()),(),True)
    sessions=ResearchAnchorSessionContext(anchor,(),(),(),(),True)
    context=ResearchAnchorSessionRelationshipContext(sessions,(),(),())
    outcomes=tuple(ForwardOutcome("EVIDENCE",TIME,11,h,15,TIME,TIME+h*STEP,TIME,TIME+(h-1)*STEP,
        11,12,13,10,1,1/11,2,1,TIME,TIME,0,0) for h in horizons)
    return assemble_research_record(mtf,context,outcomes)


def row():
    return project_research_record_with_context(record())


def test_order_duplicates_and_schemas():
    records=(record("C"),record("A"),record("B"),record("C"))
    dataset=module.build_discovery_dataset(records)
    assert [r.record for r in dataset.rows]==list(records)
    assert module.column_values(dataset,"x.geometry.level_id")==("C","A","B","C")
    assert module.column_values(dataset,"meta.evidence_end_utc")== (TIME,)*4
    assert module.column_values(dataset,"meta.outcome_anchor_id")== ("EVIDENCE",)*4
    first=dataset.rows[0]
    assert dataset.identity_field_ids==tuple(v.field_id for v in first.identity)
    assert dataset.predictor_field_ids==tuple(v.field_id for v in first.predictors)
    assert dataset.target_field_ids==tuple(v.field_id for v in first.targets)
    reordered=module.assemble_discovery_dataset((dataset.rows[2],first,first))
    assert reordered.rows[0] is dataset.rows[2]
    assert reordered.rows[1] is reordered.rows[2] is first
    assert all(s.startswith("meta.") for s in dataset.identity_field_ids)
    assert all(s.startswith("x.") and not s.startswith("y.") for s in dataset.predictor_field_ids)
    assert all(s.startswith("y.") for s in dataset.target_field_ids)


def test_empty_no_projection(monkeypatch):
    def forbidden(record):
        raise AssertionError("no projection expected")
    monkeypatch.setattr(module,"project_research_record_with_context",forbidden)
    for dataset in (module.build_discovery_dataset([]),module.assemble_discovery_dataset([])):
        assert dataset.rows==dataset.identity_field_ids==dataset.predictor_field_ids==dataset.target_field_ids==()


@pytest.mark.parametrize("field", ["identity_field_ids","predictor_field_ids","target_field_ids"])
def test_supplied_schema_mismatch(field):
    with pytest.raises(ValueError):
        replace(module.assemble_discovery_dataset((row(),)),**{field:("x.wrong",)})
    with pytest.raises(ValueError):
        replace(module.assemble_discovery_dataset(()),**{field:("x.wrong",)})


@pytest.mark.parametrize("change", ["order","missing","extra","identity","target"])
def test_row_schema_mismatch(change):
    first=row()
    if change=="order":
        second=replace(first,predictors=first.predictors[::-1])
    elif change=="missing":
        second=replace(first,predictors=first.predictors[:-1])
    elif change=="extra":
        second=replace(first,predictors=first.predictors+(DiscoveryValue("x.extra",1),))
    elif change=="identity":
        second=replace(first,identity=first.identity[:-1])
    else:
        second=replace(first,targets=first.targets[::-1])
    with pytest.raises(ValueError,match="schema mismatch"):
        module.assemble_discovery_dataset((first,second))


def test_target_horizon_schema():
    compatible=module.build_discovery_dataset((record(),record()))
    assert len(compatible.rows)==2
    with pytest.raises(ValueError,match="target schema"):
        module.build_discovery_dataset((record(),record(horizons=(1,4))))


def test_missingness_preserved_for_all_context_groups():
    source=record()
    ids=("x.mtf.ny_h4.available","x.mtf.ny_h4.direction","x.session.london.direction","x.relationship.pair.range_ratio")
    first=DiscoveryRow(source,(),tuple(DiscoveryValue(k,v) for k,v in zip(ids,(True,"BEARISH","BULLISH",2.0))),())
    second=DiscoveryRow(source,(),tuple(DiscoveryValue(k,v) for k,v in zip(ids,(False,None,None,None))),())
    dataset=module.assemble_discovery_dataset((first,second))
    assert module.column_values(dataset,ids[0])==(True,False)
    assert module.column_values(dataset,ids[1])==("BEARISH",None)
    assert module.column_values(dataset,ids[2])==("BULLISH",None)
    assert module.column_values(dataset,ids[3])==(2.0,None)


@pytest.mark.parametrize("group", ["mtf","session","relationship"])
def test_changed_context_configuration(group):
    first=replace(row(),predictors=(DiscoveryValue(f"x.{group}.original.available",True),))
    second=replace(first,predictors=(DiscoveryValue(f"x.{group}.custom.available",True),))
    with pytest.raises(ValueError):
        module.assemble_discovery_dataset((first,second))


@pytest.mark.parametrize("field,role", [("meta.evidence_end_utc","identity"),("x.anchor_kind","predictor"),("y.h8.close_change","target")])
def test_columns_and_roles(field,role):
    first=row()
    dataset=module.assemble_discovery_dataset((first,))
    assert module.field_role(dataset,field)==role
    group={"identity":"identity","predictor":"predictors","target":"targets"}[role]
    value=next(v.value for v in getattr(first,group) if v.field_id==field)
    assert module.column_values(dataset,field)==(value,)
    assert module.column_values(dataset,field)[0] is value


@pytest.mark.parametrize("value", [None,True,3,1.5,"TEXT",TIME.date(),TIME])
def test_scalar_types_unchanged(value):
    first=DiscoveryRow(record(),(DiscoveryValue("meta.scalar",value),),(),())
    assert module.column_values(module.assemble_discovery_dataset((first,)),"meta.scalar")[0] is value


def test_unknown_and_invalid_inputs():
    dataset=module.assemble_discovery_dataset((row(),))
    for function in (module.column_values,module.field_role):
        with pytest.raises(KeyError) as error:
            function(dataset,"x.unknown")
        assert error.value.args==("x.unknown",)
        with pytest.raises(TypeError):
            function(object(),"x.field")
        with pytest.raises(TypeError):
            function(dataset,123)
    with pytest.raises(TypeError):
        module.build_discovery_dataset((object(),))
    with pytest.raises(TypeError):
        module.assemble_discovery_dataset((object(),))
    with pytest.raises(TypeError):
        module.assemble_discovery_dataset((row(),object()))


def test_generators_and_exact_projection_count(monkeypatch):
    class Once:
        def __init__(self,values):
            self.values,self.calls=values,0
        def __iter__(self):
            self.calls+=1
            assert self.calls==1
            yield from self.values
    calls=[]
    def tracked(record):
        calls.append(record)
        return project_research_record_with_context(record)
    monkeypatch.setattr(module,"project_research_record_with_context",tracked)
    source=Once([record()]*100)
    dataset=module.build_discovery_dataset(source)
    assert len(calls)==100
    assert source.calls==1
    rows=Once(dataset.rows)
    rebuilt=module.assemble_discovery_dataset(rows)
    assert rows.calls==1
    assert all(a is b for a,b in zip(rebuilt.rows,dataset.rows))
    direct=Once(dataset.rows)
    module.DiscoveryDataset(direct,dataset.identity_field_ids,dataset.predictor_field_ids,dataset.target_field_ids)
    assert direct.calls==1


def test_immutable():
    dataset=module.assemble_discovery_dataset((row(),))
    with pytest.raises(FrozenInstanceError):
        dataset.rows=()
