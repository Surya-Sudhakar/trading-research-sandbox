"""Append-only identity registry, independent of execution plugins."""
from sandbox.research.errors import ResearchError
from .models import StrategyDefinition, StrategyFamily, StrategyVariant


class StrategyIdentityRegistry:
    """IDs are unique within their parent; families are globally unique.

    Records cannot be replaced, including frozen baselines; new variants are
    allowed. Inputs and outputs are deep snapshots, so caller mutations of
    nested metadata or parameters cannot change registered records.
    """

    def __init__(self):
        self._families: dict[str, StrategyFamily] = {}
        self._strategies: dict[tuple[str, str], StrategyDefinition] = {}
        self._variants: dict[tuple[str, str, str], StrategyVariant] = {}

    @staticmethod
    def _register(records, key, value):
        if key in records:
            raise ResearchError(f"DUPLICATE_STRATEGY_IDENTITY: {key}")
        # Revalidate even model_copy/model_construct inputs before storing.
        stored = type(value).model_validate(value.model_dump()).model_copy(deep=True)
        records[key] = stored
        return stored.model_copy(deep=True)

    @staticmethod
    def _get(records, key):
        if key not in records:
            raise ResearchError(f"STRATEGY_IDENTITY_NOT_FOUND: {key}")
        return records[key].model_copy(deep=True)

    def register_family(self, family: StrategyFamily) -> StrategyFamily:
        return self._register(self._families, family.family_id, family)

    def register_strategy(self, strategy: StrategyDefinition) -> StrategyDefinition:
        self.get_family(strategy.family_id)
        return self._register(self._strategies, (strategy.family_id, strategy.strategy_id), strategy)

    def register_variant(self, variant: StrategyVariant) -> StrategyVariant:
        self.get_strategy(variant.family_id, variant.strategy_id)
        if variant.parent_variant_id is not None:
            self.get_variant(variant.family_id, variant.strategy_id, variant.parent_variant_id)
        return self._register(self._variants, (variant.family_id, variant.strategy_id, variant.variant_id), variant)

    def get_family(self, family_id: str) -> StrategyFamily:
        return self._get(self._families, family_id)

    def get_strategy(self, family_id: str, strategy_id: str) -> StrategyDefinition:
        return self._get(self._strategies, (family_id, strategy_id))

    def get_variant(self, family_id: str, strategy_id: str, variant_id: str) -> StrategyVariant:
        return self._get(self._variants, (family_id, strategy_id, variant_id))

    def list_families(self) -> tuple[StrategyFamily, ...]:
        return tuple(self.get_family(key) for key in sorted(self._families))

    def list_strategies(self, family_id: str) -> tuple[StrategyDefinition, ...]:
        self.get_family(family_id)
        return tuple(self.get_strategy(*key) for key in sorted(self._strategies) if key[0] == family_id)

    def list_variants(self, family_id: str, strategy_id: str) -> tuple[StrategyVariant, ...]:
        self.get_strategy(family_id, strategy_id)
        return tuple(self.get_variant(*key) for key in sorted(self._variants) if key[:2] == (family_id, strategy_id))
