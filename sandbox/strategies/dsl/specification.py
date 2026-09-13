"""Compose existing declarative condition and execution objects."""
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .models import MachineId
from .conditions import ConditionExpression
from .execution import EntrySpec, StopSpec, TargetSpec


class SpecificationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False,
                              revalidate_instances="always")


class TradeRule(SpecificationModel):
    rule_id: MachineId
    condition: ConditionExpression
    entry: EntrySpec
    stop: StopSpec | None = None
    target: TargetSpec | None = None


class StrategySpec(SpecificationModel):
    family_id: MachineId
    strategy_id: MachineId
    variant_id: MachineId
    rules: tuple[TradeRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_rule_ids(self) -> Self:
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("duplicate rule_id within StrategySpec")
        return self
