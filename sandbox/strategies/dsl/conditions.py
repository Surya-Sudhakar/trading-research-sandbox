"""Declarative condition AST. This module does not evaluate expressions.

References are open-ended names, not a catalog of available features. Timeframe
and context use the existing machine identifier convention; no timeframe catalog
or session semantics are imposed. Pydantic rejects cyclic/excessively recursive
input with validation errors rather than following it indefinitely.
"""
from enum import StrEnum
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, TypeAdapter

from .models import MachineId

NonBlank = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]


class ConditionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False,
                              revalidate_instances="always")


class ValueReference(ConditionModel):
    namespace: NonBlank
    key: NonBlank
    timeframe: MachineId | None = None
    context: MachineId | None = None


class ComparisonOperator(StrEnum):
    EQ = "EQ"
    NE = "NE"
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"
    IN = "IN"
    NOT_IN = "NOT_IN"


class ComparisonCondition(ConditionModel):
    kind: Literal["comparison"] = "comparison"
    reference: ValueReference
    operator: ComparisonOperator
    value: JsonValue


class AndCondition(ConditionModel):
    kind: Literal["and"] = "and"
    children: tuple["ConditionExpression", ...] = Field(min_length=1)


class OrCondition(ConditionModel):
    kind: Literal["or"] = "or"
    children: tuple["ConditionExpression", ...] = Field(min_length=1)


class NotCondition(ConditionModel):
    kind: Literal["not"] = "not"
    children: tuple["ConditionExpression", ...] = Field(min_length=1, max_length=1)


ConditionExpression = Annotated[
    ComparisonCondition | AndCondition | OrCondition | NotCondition,
    Field(discriminator="kind"),
]

for _node in (AndCondition, OrCondition, NotCondition):
    _node.model_rebuild()

_expression_adapter = TypeAdapter(ConditionExpression)


def parse_condition(data: object) -> ConditionExpression:
    """Validate a tagged tree (or model), without interpreting its values."""
    return _expression_adapter.validate_python(data)


def canonical_condition_json(expression: ConditionExpression) -> str:
    """Canonical UTF-8-compatible JSON text; child/list order is significant.

    Includes defaults/nulls, sorts mapping keys, disallows nonfinite numbers,
    and uses compact separators. No algebraic normalization is performed.
    Revalidation also catches invalid mutations of nested JSON data.
    """
    validated = parse_condition(expression)
    return json.dumps(validated.model_dump(mode="json"), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False, allow_nan=False)
