"""Generic strategy identities with no execution semantics."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

MachineId = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z0-9_]+$")]


class IdentityModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()


class StrategyFamily(IdentityModel):
    family_id: MachineId
    name: str = Field(min_length=1)
    description: str


class StrategyDefinition(IdentityModel):
    family_id: MachineId
    strategy_id: MachineId
    name: str = Field(min_length=1)
    description: str
    baseline_frozen: bool = False


class StrategyVariant(IdentityModel):
    family_id: MachineId
    strategy_id: MachineId
    variant_id: MachineId
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    parent_variant_id: MachineId | None = None
