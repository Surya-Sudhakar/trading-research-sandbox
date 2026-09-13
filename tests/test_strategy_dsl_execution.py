import pytest
from pydantic import TypeAdapter, ValidationError

from sandbox.strategies.dsl import (
    ValueReference, PositionSide, MarketEntry, LimitEntry, StopEntry, EntrySpec,
    ReferenceStop, FixedPipsStop, AtrStop, StopSpec,
    ReferenceTarget, FixedPipsTarget, AtrTarget, RiskMultipleTarget, TargetSpec,
)

PRICE = ValueReference(namespace="price", key="level", timeframe="M15")
ATR = ValueReference(namespace="feature", key="atr", timeframe="H4")
GEOMETRY = ValueReference(namespace="geometry", key="bb_projection_distance")

CASES = [
    (EntrySpec, MarketEntry, {"kind": "market", "side": "LONG"}),
    (EntrySpec, MarketEntry, {"kind": "market", "side": "SHORT"}),
    (EntrySpec, LimitEntry, {"kind": "limit", "side": "LONG", "price_reference": PRICE}),
    (EntrySpec, StopEntry, {"kind": "stop", "side": "SHORT", "price_reference": PRICE}),
    (StopSpec, ReferenceStop, {"kind": "reference", "reference": GEOMETRY}),
    (StopSpec, FixedPipsStop, {"kind": "fixed_pips", "pips": 12.5}),
    (StopSpec, AtrStop, {"kind": "atr_multiple", "atr_reference": ATR, "multiple": 1.5}),
    (TargetSpec, ReferenceTarget, {"kind": "reference", "reference": PRICE}),
    (TargetSpec, FixedPipsTarget, {"kind": "fixed_pips", "pips": 20.0}),
    (TargetSpec, AtrTarget, {"kind": "atr_multiple", "atr_reference": ATR, "multiple": 2.0}),
    (TargetSpec, RiskMultipleTarget, {"kind": "r_multiple", "multiple": 3.0}),
]


@pytest.mark.parametrize("spec,model,data", CASES)
def test_declarative_variants(spec, model, data):
    node = TypeAdapter(spec).validate_python(data)
    assert type(node) is model
    assert node == model(**data)
    for field, value in data.items():
        assert getattr(node, field) == value
    if isinstance(node, MarketEntry):
        assert isinstance(node.side, PositionSide)


@pytest.mark.parametrize("model,field,extra", [
    (FixedPipsStop, "pips", {}), (FixedPipsTarget, "pips", {}),
    (AtrStop, "multiple", {"atr_reference": ATR}),
    (AtrTarget, "multiple", {"atr_reference": ATR}),
    (RiskMultipleTarget, "multiple", {}),
])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_reject_nonpositive_or_nonfinite(model, field, extra, value):
    with pytest.raises(ValidationError):
        model(**{field: value}, **extra)


@pytest.mark.parametrize("model,extra", [(MarketEntry, {}),
    (LimitEntry, {"price_reference": PRICE}), (StopEntry, {"price_reference": PRICE})])
def test_unknown_side(model, extra):
    with pytest.raises(ValidationError):
        model(side="BUY", **extra)


@pytest.mark.parametrize("spec", [EntrySpec, StopSpec, TargetSpec])
@pytest.mark.parametrize("data", [{"kind": "unknown"}, {}])
def test_unknown_or_missing_kind(spec, data):
    with pytest.raises(ValidationError):
        TypeAdapter(spec).validate_python(data)


@pytest.mark.parametrize("spec,model,data", CASES)
def test_models_frozen_and_extra_forbidden(spec, model, data):
    node = model(**data)
    with pytest.raises(ValidationError):
        node.kind = data["kind"]
    with pytest.raises(ValidationError):
        model(**data, unexpected=True)
    with pytest.raises(ValidationError):
        model(**(data | {"kind": "unknown"}))


@pytest.mark.parametrize("model,field", [(LimitEntry, "price_reference"),
    (StopEntry, "price_reference"), (ReferenceStop, "reference"),
    (ReferenceTarget, "reference"), (AtrStop, "atr_reference"), (AtrTarget, "atr_reference")])
def test_reference_is_required(model, field):
    fields = {"side": "LONG"} if model in (LimitEntry, StopEntry) else {}
    if model in (AtrStop, AtrTarget):
        fields["multiple"] = 1.0
    with pytest.raises(ValidationError):
        model(**fields)
    with pytest.raises(ValidationError):
        model(**fields, **{field: {"namespace": "", "key": "level"}})


def test_geometry_uses_generic_models_and_frozen_reference():
    stop = ReferenceStop(reference=GEOMETRY)
    assert type(stop) is ReferenceStop
    assert type(stop.reference) is ValueReference
    assert stop.reference.namespace == "geometry"
    assert stop.reference.key == "bb_projection_distance"
    with pytest.raises(ValidationError):
        stop.reference.key = "changed"
