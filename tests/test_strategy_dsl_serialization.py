import hashlib
import json
import re

import pytest
from pydantic import ValidationError

from sandbox.strategies.dsl import (
    StrategySpec, TradeRule, ValueReference, ComparisonCondition, AndCondition,
    MarketEntry, FixedPipsStop, RiskMultipleTarget,
    STRATEGY_SPEC_CANONICAL_VERSION, canonical_strategy_spec_json, strategy_spec_fingerprint,
)


def condition(value=True):
    return ComparisonCondition(reference=ValueReference(namespace="event", key="BB_LEVEL_TOUCH"),
                               operator="EQ", value=value)


def rule(**changes):
    return TradeRule(**({"rule_id": "SHORT_TOUCH", "condition": condition(),
        "entry": MarketEntry(side="SHORT"), "stop": FixedPipsStop(pips=10),
        "target": RiskMultipleTarget(multiple=2)} | changes))


def spec(**changes):
    return StrategySpec(**({"family_id": "BIG_BROTHER", "strategy_id": "VERSION_0",
        "variant_id": "ORIGINAL", "rules": (rule(),)} | changes))


def test_repeated_serialization_identical():
    node = spec()
    assert canonical_strategy_spec_json(node) == canonical_strategy_spec_json(node)


def test_repeated_fingerprint_identical():
    node = spec()
    assert strategy_spec_fingerprint(node) == strategy_spec_fingerprint(node)


def test_reconstructed_spec_same_fingerprint():
    node = spec()
    assert strategy_spec_fingerprint(StrategySpec(**node.model_dump())) == strategy_spec_fingerprint(node)


def test_dictionary_order_irrelevant():
    a = spec(rules=(rule(condition=condition({"z": [1, 2], "a": {"y": True, "b": None}})),))
    b = spec(rules=(rule(condition=condition({"a": {"b": None, "y": True}, "z": [1, 2]})),))
    assert canonical_strategy_spec_json(a) == canonical_strategy_spec_json(b)
    assert strategy_spec_fingerprint(a) == strategy_spec_fingerprint(b)


@pytest.mark.parametrize("changes", [
    {"condition": condition(False)}, {"entry": MarketEntry(side="LONG")},
    {"stop": FixedPipsStop(pips=11)}, {"target": RiskMultipleTarget(multiple=2.5)},
    {"rule_id": "ANOTHER_RULE"}, {"stop": None}, {"target": None},
])
def test_rule_changes_change_fingerprint(changes):
    assert strategy_spec_fingerprint(spec(rules=(rule(**changes),))) != strategy_spec_fingerprint(spec())


@pytest.mark.parametrize("field", ["family_id", "strategy_id", "variant_id"])
def test_identity_changes_change_fingerprint(field):
    assert strategy_spec_fingerprint(spec(**{field: "CHANGED"})) != strategy_spec_fingerprint(spec())


def test_rule_order_significant():
    a, b = rule(), rule(rule_id="LONG_TOUCH", entry=MarketEntry(side="LONG"))
    assert strategy_spec_fingerprint(spec(rules=(a, b))) != strategy_spec_fingerprint(spec(rules=(b, a)))


def test_and_child_order_significant():
    a, b = condition(True), condition(False)
    first = spec(rules=(rule(condition=AndCondition(children=(a, b))),))
    second = spec(rules=(rule(condition=AndCondition(children=(b, a))),))
    assert strategy_spec_fingerprint(first) != strategy_spec_fingerprint(second)


def test_value_list_order_significant():
    first = spec(rules=(rule(condition=condition([1, 2])),))
    second = spec(rules=(rule(condition=condition([2, 1])),))
    assert strategy_spec_fingerprint(first) != strategy_spec_fingerprint(second)


def test_sha256_format_and_exact_bytes():
    node = spec()
    fingerprint = strategy_spec_fingerprint(node)
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert fingerprint == hashlib.sha256(canonical_strategy_spec_json(node).encode("utf-8")).hexdigest()


def test_exact_envelope_contains_only_version_and_complete_spec():
    node = spec()
    output = canonical_strategy_spec_json(node)
    assert STRATEGY_SPEC_CANONICAL_VERSION == "STRATEGY_SPEC_V1"
    assert '"schema_version":"STRATEGY_SPEC_V1"' in output
    expected = {"schema_version": "STRATEGY_SPEC_V1", "spec": node.model_dump(mode="json")}
    assert json.loads(output) == expected
    assert output == json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("invalid", [object(), float("nan"), float("inf"), -float("inf")])
def test_invalid_nested_json_mutation_detected(invalid):
    node = spec(rules=(rule(condition=AndCondition(children=(condition({"nested": {"items": []}}),))),))
    node.rules[0].condition.children[0].value["nested"]["items"].append(invalid)
    with pytest.raises(ValidationError):
        canonical_strategy_spec_json(node)
    with pytest.raises(ValidationError):
        strategy_spec_fingerprint(node)


def test_invalid_nested_execution_revalidated():
    invalid_rule = rule().model_copy(update={"stop": FixedPipsStop.model_construct(pips=-1)})
    invalid_spec = spec().model_copy(update={"rules": (invalid_rule,)})
    with pytest.raises(ValidationError):
        canonical_strategy_spec_json(invalid_spec)


def test_duplicate_rules_revalidated():
    invalid = spec().model_copy(update={"rules": (rule(), rule())})
    with pytest.raises(ValidationError, match="duplicate rule_id"):
        canonical_strategy_spec_json(invalid)


def test_no_environment_or_runtime_fields(monkeypatch):
    node = spec()
    before = canonical_strategy_spec_json(node)
    monkeypatch.setenv("HOSTNAME", "unrelated-host")
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    assert canonical_strategy_spec_json(node) == before
    data = json.loads(before)
    assert set(data) == {"schema_version", "spec"}
    assert set(data["spec"]) == {"family_id", "strategy_id", "variant_id", "rules"}
    assert set(data["spec"]["rules"][0]) == {"rule_id", "condition", "entry", "stop", "target"}


def test_utf8_and_optional_fields():
    node = spec(rules=(rule(condition=condition("caf\u00e9 \u6771\u4eac"), stop=None, target=None),))
    output = canonical_strategy_spec_json(node)
    assert output.encode("utf-8").decode("utf-8") == output
    assert "caf\u00e9 \u6771\u4eac" in output
    assert json.loads(output)["spec"]["rules"][0]["stop"] is None
    assert json.loads(output)["spec"]["rules"][0]["target"] is None
