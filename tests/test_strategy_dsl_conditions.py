import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from sandbox.strategies.dsl import (
    ValueReference, ComparisonOperator, ComparisonCondition, AndCondition,
    OrCondition, NotCondition, parse_condition, canonical_condition_json,
)


def comparison(namespace="feature", key="atr_percentile_100", value=30, operator="LT", **reference):
    return ComparisonCondition(reference=ValueReference(namespace=namespace, key=key, **reference),
                               operator=operator, value=value)


def test_simple_feature_comparison():
    node = comparison()
    assert node.reference.namespace == "feature"
    assert node.reference.key == "atr_percentile_100"
    assert node.operator == ComparisonOperator.LT
    assert node.value == 30


@pytest.mark.parametrize("namespace,key,timeframe,context,value", [
    ("state", "trend", "D1", None, "BULLISH"),
    ("state", "volatility", "H4", None, "CONTRACTING"),
    ("event", "bb_level_touch", None, "VERSION_0", True),
    ("session", "london_direction", None, "LONDON", "BULLISH"),
    ("session", "reclaim", None, "NEW_YORK", True),
    ("geometry", "bb_projection_distance", "M15", None, 2.5),
    ("price", "zone", None, None, {"low": 1, "high": 2}),
    ("machine_discovered", "future_feature_0001", "M15", None, None),
])
def test_generic_references(namespace, key, timeframe, context, value):
    node = comparison(namespace, key, value, "EQ", timeframe=timeframe, context=context)
    assert parse_condition(json.loads(canonical_condition_json(node))) == node


@pytest.mark.parametrize("operator", list(ComparisonOperator))
def test_all_operators(operator):
    value = ["BULLISH", "BEARISH"] if operator in (ComparisonOperator.IN, ComparisonOperator.NOT_IN) else 30
    assert comparison(operator=operator, value=value).operator == operator


@pytest.mark.parametrize("node_type", [AndCondition, OrCondition, NotCondition])
def test_logical_nodes(node_type):
    node = node_type(children=(comparison(),))
    assert parse_condition(node.model_dump()) == node


def test_nested_expression():
    node = AndCondition(children=(
        comparison("state", "trend", "BULLISH", "EQ", timeframe="D1"),
        comparison("state", "volatility", "CONTRACTING", "EQ", timeframe="H4"),
        OrCondition(children=(comparison("event", "BB_TOUCH", True, "EQ"),
                              NotCondition(children=(comparison("event", "BB_RECLAIM", True, "EQ"),)))),
    ))
    assert parse_condition(json.loads(canonical_condition_json(node))) == node


@pytest.mark.parametrize("field,value", [("namespace", ""), ("namespace", "  "), ("key", ""),
    ("key", "\n"), ("timeframe", "d1"), ("timeframe", "D1\n"), ("context", "New York"), ("context", "")])
def test_invalid_reference_fields(field, value):
    with pytest.raises(ValidationError):
        ValueReference(**({"namespace": "feature", "key": "test"} | {field: value}))


@pytest.mark.parametrize("operator", ["UNKNOWN", ">", "eq", "__import__('os')"])
def test_invalid_operators(operator):
    with pytest.raises(ValidationError):
        comparison(operator=operator)


@pytest.mark.parametrize("node_type,children", [(AndCondition, ()), (OrCondition, ()),
    (NotCondition, ()), (NotCondition, (comparison(), comparison()))])
def test_invalid_arity(node_type, children):
    with pytest.raises(ValidationError):
        node_type(children=children)


@pytest.mark.parametrize("data", [{}, {"kind": "unknown"}, {"kind": "comparison"},
    {"kind": "and", "children": ["x > 1"]}, {"kind": "not", "children": None},
    {"kind": "or", "children": [], "executable": "anything"}])
def test_malformed_nodes(data):
    with pytest.raises(ValidationError):
        parse_condition(data)


@pytest.mark.parametrize("value", [object(), lambda: True, datetime(2024, 1, 1), {1, 2},
    float("nan"), float("inf"), {"nested": [float("inf")]}, {"nested": object()}])
def test_unsafe_values(value):
    with pytest.raises(ValidationError):
        comparison(value=value)


def test_canonical_mapping_order_and_ordered_children():
    a = comparison(value={"z": [2, 1], "a": {"y": True, "b": None}})
    b = comparison(value={"a": {"b": None, "y": True}, "z": [2, 1]})
    assert canonical_condition_json(a) == canonical_condition_json(b)
    assert canonical_condition_json(a) == canonical_condition_json(parse_condition(a.model_dump()))
    c = comparison(value=10)
    assert canonical_condition_json(AndCondition(children=(a, c))) != canonical_condition_json(AndCondition(children=(c, a)))
    assert canonical_condition_json(comparison(value=[1, 2])) != canonical_condition_json(comparison(value=[2, 1]))
    assert canonical_condition_json(comparison()) == '{"kind":"comparison","operator":"LT","reference":{"context":null,"key":"atr_percentile_100","namespace":"feature","timeframe":null},"value":30}'


def test_deep_valid_tree():
    data = comparison().model_dump()
    for _ in range(60):
        data = {"kind": "not", "children": [data]}
    node = parse_condition(data)
    assert json.loads(canonical_condition_json(node)) == json.loads(json.dumps(data))


def test_cycle_rejected():
    data = {"kind": "and", "children": []}
    data["children"].append(data)
    with pytest.raises(ValidationError):
        parse_condition(data)


def test_serialization_revalidates_mutated_values():
    node = comparison(value={"items": []})
    node.value["items"].append(object())
    with pytest.raises(ValidationError):
        canonical_condition_json(node)


def test_code_like_strings_remain_literal_data():
    text = "__import__('os').system('anything')"
    assert json.loads(canonical_condition_json(comparison(value=text)))["value"] == text
