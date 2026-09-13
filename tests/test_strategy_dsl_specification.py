import pytest
from pydantic import ValidationError
from sandbox.strategies.dsl import (
    SpecificationModel, TradeRule, StrategySpec, ValueReference,
    ComparisonCondition, AndCondition, OrCondition, NotCondition,
    MarketEntry, PositionSide, ReferenceStop, RiskMultipleTarget,
)


def rule(**changes):
    return TradeRule(**({
        "rule_id": "SHORT_TOUCH",
        "condition": ComparisonCondition(
            reference=ValueReference(namespace="event", key="BB_LEVEL_TOUCH"),
            operator="EQ", value=True),
        "entry": MarketEntry(side="SHORT"),
        "stop": ReferenceStop(reference=ValueReference(namespace="geometry", key="bb_stop_level")),
        "target": RiskMultipleTarget(multiple=2.0),
    } | changes))


def spec(**changes):
    return StrategySpec(**({"family_id": "BIG_BROTHER", "strategy_id": "VERSION_0",
        "variant_id": "ORIGINAL", "rules": (rule(),)} | changes))


def test_complete_rule():
    node = rule()
    assert node.rule_id == "SHORT_TOUCH"
    assert node.condition.reference.key == "BB_LEVEL_TOUCH"
    assert node.condition.operator == "EQ"
    assert node.condition.value is True
    assert node.entry.side is PositionSide.SHORT
    assert node.stop.reference.namespace == "geometry"
    assert node.target.multiple == 2.0


def test_one_rule_and_generic_identity():
    node = spec()
    assert (node.family_id, node.strategy_id, node.variant_id) == ("BIG_BROTHER", "VERSION_0", "ORIGINAL")
    assert node.rules == (rule(),)
    assert type(node) is StrategySpec
    assert type(node.rules[0]) is TradeRule
    assert type(node.rules[0].stop) is ReferenceStop
    assert type(node.rules[0].stop.reference) is ValueReference


def test_multiple_long_and_short_rules():
    node = spec(rules=(rule(), rule(rule_id="LONG_TOUCH", entry=MarketEntry(side="LONG"))))
    assert [r.rule_id for r in node.rules] == ["SHORT_TOUCH", "LONG_TOUCH"]
    assert [r.entry.side for r in node.rules] == [PositionSide.SHORT, PositionSide.LONG]


@pytest.mark.parametrize("stop_none,target_none", [(True, False), (False, True), (True, True)])
def test_optional_exits(stop_none, target_none):
    changes = {}
    if stop_none:
        changes["stop"] = None
    if target_none:
        changes["target"] = None
    node = rule(**changes)
    assert (node.stop is None) == stop_none
    assert (node.target is None) == target_none
    assert spec(rules=(node,)).rules == (node,)


def test_exits_default_to_none():
    original = rule()
    node = TradeRule(rule_id="NO_EXITS", condition=original.condition, entry=original.entry)
    assert node.stop is None
    assert node.target is None


def test_reference_stop_and_r_multiple_target():
    node = spec().rules[0]
    assert isinstance(node.stop, ReferenceStop)
    assert node.stop.reference.key == "bb_stop_level"
    assert isinstance(node.target, RiskMultipleTarget)
    assert node.target.multiple == 2.0


def test_empty_rules_rejected():
    with pytest.raises(ValidationError):
        spec(rules=())


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate rule_id"):
        spec(rules=(rule(), rule(entry=MarketEntry(side="LONG"))))


@pytest.mark.parametrize("field", ["rule_id", "family_id", "strategy_id", "variant_id"])
@pytest.mark.parametrize("bad", ["", "lowercase", "HAS SPACE", "HAS-DASH", "A\n", 123])
def test_invalid_ids(field, bad):
    with pytest.raises(ValidationError):
        (rule if field == "rule_id" else spec)(**{field: bad})


def test_nested_ast_preserved():
    a = rule().condition
    b = ComparisonCondition(reference=ValueReference(namespace="state", key="trend", timeframe="D1"),
                            operator="EQ", value="BEARISH")
    tree = AndCondition(children=(b, OrCondition(children=(a, NotCondition(children=(b,))))))
    node = spec(rules=(rule(condition=tree),))
    assert node.rules[0].condition == tree
    assert isinstance(node.rules[0].condition.children[1], OrCondition)
    assert isinstance(node.rules[0].condition.children[1].children[1], NotCondition)


@pytest.mark.parametrize("node,field,value", [(rule(), "rule_id", "CHANGED"),
    (spec(), "rules", ()), (spec(), "variant_id", "CHANGED")])
def test_frozen(node, field, value):
    with pytest.raises(ValidationError):
        setattr(node, field, value)


@pytest.mark.parametrize("factory", [rule, spec])
def test_extra_fields(factory):
    with pytest.raises(ValidationError):
        factory(unexpected=True)


def test_configuration():
    for model in (SpecificationModel, TradeRule, StrategySpec):
        assert model.model_config["frozen"] is True
        assert model.model_config["extra"] == "forbid"
        assert model.model_config["allow_inf_nan"] is False
        assert model.model_config["revalidate_instances"] == "always"


def test_nested_tagged_data():
    node = TradeRule(rule_id="DATA_RULE", condition={
        "kind": "comparison", "reference": {"namespace": "event", "key": "touch"},
        "operator": "EQ", "value": True}, entry={"kind": "market", "side": "SHORT"},
        stop={"kind": "reference", "reference": {"namespace": "geometry", "key": "level"}},
        target={"kind": "r_multiple", "multiple": 2.0})
    assert isinstance(node.condition, ComparisonCondition)
    assert isinstance(node.entry, MarketEntry)
    assert isinstance(node.stop, ReferenceStop)
    assert isinstance(node.target, RiskMultipleTarget)


def test_invalid_instance_revalidated():
    invalid = rule().model_copy(update={"rule_id": "invalid"})
    with pytest.raises(ValidationError):
        spec(rules=(invalid,))
