"""Generic declarative execution specifications; no price or order evaluation."""
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .conditions import ValueReference

PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ExecutionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False,
                              revalidate_instances="always")


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class MarketEntry(ExecutionModel):
    kind: Literal["market"] = "market"
    side: PositionSide


class LimitEntry(ExecutionModel):
    kind: Literal["limit"] = "limit"
    side: PositionSide
    price_reference: ValueReference


class StopEntry(ExecutionModel):
    kind: Literal["stop"] = "stop"
    side: PositionSide
    price_reference: ValueReference


EntrySpec = Annotated[MarketEntry | LimitEntry | StopEntry, Field(discriminator="kind")]


class ReferenceStop(ExecutionModel):
    kind: Literal["reference"] = "reference"
    reference: ValueReference


class FixedPipsStop(ExecutionModel):
    kind: Literal["fixed_pips"] = "fixed_pips"
    pips: PositiveFiniteFloat


class AtrStop(ExecutionModel):
    kind: Literal["atr_multiple"] = "atr_multiple"
    atr_reference: ValueReference
    multiple: PositiveFiniteFloat


StopSpec = Annotated[ReferenceStop | FixedPipsStop | AtrStop, Field(discriminator="kind")]


class ReferenceTarget(ExecutionModel):
    kind: Literal["reference"] = "reference"
    reference: ValueReference


class FixedPipsTarget(ExecutionModel):
    kind: Literal["fixed_pips"] = "fixed_pips"
    pips: PositiveFiniteFloat


class AtrTarget(ExecutionModel):
    kind: Literal["atr_multiple"] = "atr_multiple"
    atr_reference: ValueReference
    multiple: PositiveFiniteFloat


class RiskMultipleTarget(ExecutionModel):
    kind: Literal["r_multiple"] = "r_multiple"
    multiple: PositiveFiniteFloat


TargetSpec = Annotated[
    ReferenceTarget | FixedPipsTarget | AtrTarget | RiskMultipleTarget,
    Field(discriminator="kind"),
]
