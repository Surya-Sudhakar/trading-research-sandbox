"""Strategy DSL identity and declarative condition foundation."""
from .models import StrategyDefinition, StrategyFamily, StrategyVariant
from .registry import StrategyIdentityRegistry
from .conditions import (
    ValueReference, ComparisonOperator, ComparisonCondition, AndCondition,
    OrCondition, NotCondition, ConditionExpression, parse_condition,
    canonical_condition_json,
)

__all__ = [
    "StrategyFamily", "StrategyDefinition", "StrategyVariant", "StrategyIdentityRegistry",
    "ValueReference", "ComparisonOperator", "ComparisonCondition", "AndCondition",
    "OrCondition", "NotCondition", "ConditionExpression", "parse_condition",
    "canonical_condition_json",
]

from .execution import (
    PositionSide, MarketEntry, LimitEntry, StopEntry, EntrySpec,
    ReferenceStop, FixedPipsStop, AtrStop, StopSpec,
    ReferenceTarget, FixedPipsTarget, AtrTarget, RiskMultipleTarget, TargetSpec,
)

__all__ += [
    "PositionSide", "MarketEntry", "LimitEntry", "StopEntry", "EntrySpec",
    "ReferenceStop", "FixedPipsStop", "AtrStop", "StopSpec",
    "ReferenceTarget", "FixedPipsTarget", "AtrTarget", "RiskMultipleTarget", "TargetSpec",
]

from .specification import SpecificationModel, TradeRule, StrategySpec

__all__ += ["SpecificationModel", "TradeRule", "StrategySpec"]

from .serialization import (
    STRATEGY_SPEC_CANONICAL_VERSION, canonical_strategy_spec_json, strategy_spec_fingerprint,
)

__all__ += [
    "STRATEGY_SPEC_CANONICAL_VERSION", "canonical_strategy_spec_json", "strategy_spec_fingerprint",
]
