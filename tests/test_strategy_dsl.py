import pytest
from pydantic import ValidationError

from sandbox.research.errors import ResearchError
from sandbox.strategies.dsl import (
    StrategyFamily, StrategyDefinition, StrategyVariant, StrategyIdentityRegistry,
)


def family(**changes):
    return StrategyFamily(**({"family_id": "BIG_BROTHER", "name": "Big Brother", "description": "Family"} | changes))


def definition(strategy_id="BB_ORIGINAL", **changes):
    return StrategyDefinition(**({"family_id": "BIG_BROTHER", "strategy_id": strategy_id,
        "name": strategy_id, "description": "Definition", "baseline_frozen": True,
        "metadata": {"history": {"labels": ["original"]}}, "tags": ("baseline",)} | changes))


def variant(strategy_id="BB_ORIGINAL", variant_id="ORIGINAL", **changes):
    return StrategyVariant(**({"family_id": "BIG_BROTHER", "strategy_id": strategy_id,
        "variant_id": variant_id, "parameters": {"window": [10]}} | changes))


@pytest.fixture
def registry():
    result = StrategyIdentityRegistry()
    result.register_family(family())
    for strategy_id in ("BB_ORIGINAL", "VERSION_0"):
        result.register_strategy(definition(strategy_id))
        result.register_variant(variant(strategy_id))
    return result


def test_three_level_example_and_additional_variants(registry):
    assert [f.family_id for f in registry.list_families()] == ["BIG_BROTHER"]
    assert [s.strategy_id for s in registry.list_strategies("BIG_BROTHER")] == ["BB_ORIGINAL", "VERSION_0"]
    with pytest.raises(ResearchError, match="NOT_FOUND"):
        registry.get_family("VERSION_0")
    for strategy_id in ("BB_ORIGINAL", "VERSION_0"):
        assert registry.get_variant("BIG_BROTHER", strategy_id, "ORIGINAL").strategy_id == strategy_id
        for variant_id in ("V0002", "V0001"):
            registry.register_variant(variant(strategy_id, variant_id, parent_variant_id="ORIGINAL"))
        assert [v.variant_id for v in registry.list_variants("BIG_BROTHER", strategy_id)] == ["ORIGINAL", "V0001", "V0002"]


@pytest.mark.parametrize("method,value", [
    ("register_family", family()), ("register_strategy", definition()), ("register_variant", variant()),
])
def test_duplicates_rejected(registry, method, value):
    with pytest.raises(ResearchError, match="DUPLICATE"):
        getattr(registry, method)(value)


@pytest.mark.parametrize("value", [definition(family_id="MISSING")])
def test_missing_family(registry, value):
    with pytest.raises(ResearchError, match="NOT_FOUND"):
        registry.register_strategy(value)


@pytest.mark.parametrize("changes", [{"strategy_id": "MISSING"}, {"family_id": "MISSING"}, {"parent_variant_id": "MISSING"}, {"variant_id": "SELF", "parent_variant_id": "SELF"}])
def test_missing_variant_parents(registry, changes):
    with pytest.raises(ResearchError, match="NOT_FOUND"):
        registry.register_variant(variant(**changes))


def test_variant_parent_must_belong_to_same_definition(registry):
    registry.register_variant(variant(variant_id="V0001"))
    with pytest.raises(ResearchError, match="NOT_FOUND"):
        registry.register_variant(variant("VERSION_0", "V0002", parent_variant_id="V0001"))


def test_frozen_baseline_survives_input_and_output_mutation(registry):
    original = definition("FROZEN")
    expected = original.model_dump()
    returned = registry.register_strategy(original)
    original.metadata["history"]["labels"].append("input mutation")
    returned.metadata.clear()
    read = registry.get_strategy("BIG_BROTHER", "FROZEN")
    with pytest.raises(ValidationError):
        read.baseline_frozen = False
    read.metadata.clear()
    registry.list_strategies("BIG_BROTHER")[-1].metadata.clear()
    assert registry.get_strategy("BIG_BROTHER", "FROZEN").model_dump() == expected
    with pytest.raises(ResearchError, match="DUPLICATE"):
        registry.register_strategy(definition("FROZEN", baseline_frozen=False))
    registry.register_variant(variant("FROZEN"))
    registry.register_variant(variant("FROZEN", "V0001", parent_variant_id="ORIGINAL"))
    assert registry.get_strategy("BIG_BROTHER", "FROZEN").model_dump() == expected


def test_family_and_variant_snapshots_are_isolated(registry):
    f = registry.get_family("BIG_BROTHER")
    f.metadata["new"] = True
    assert registry.get_family("BIG_BROTHER").metadata == {}
    v = variant(variant_id="V0001")
    returned = registry.register_variant(v)
    v.parameters["window"].append(20)
    returned.parameters.clear()
    registry.list_variants("BIG_BROTHER", "BB_ORIGINAL")[1].parameters.clear()
    assert registry.get_variant("BIG_BROTHER", "BB_ORIGINAL", "V0001").parameters == {"window": [10]}


@pytest.mark.parametrize("bad", ["", "lowercase", "HAS SPACE", "HAS-DASH", "A\n", "É", "A.B", 123])
@pytest.mark.parametrize("factory,field", [(family, "family_id"), (definition, "family_id"), (definition, "strategy_id"), (variant, "family_id"), (variant, "strategy_id"), (variant, "variant_id"), (variant, "parent_variant_id")])
def test_invalid_machine_ids(factory, field, bad):
    with pytest.raises(ValidationError):
        factory(**{field: bad})


@pytest.mark.parametrize("family_id", ["VOLATILITY_EXPANSION", "BREAK_RETEST", "MACHINE_DISCOVERED_0001"])
def test_generic_families_and_scoped_strategy_ids(registry, family_id):
    registry.register_family(family(family_id=family_id))
    registry.register_strategy(definition(family_id=family_id))
    registry.register_variant(variant(family_id=family_id))
    assert registry.get_variant(family_id, "BB_ORIGINAL", "ORIGINAL").family_id == family_id
