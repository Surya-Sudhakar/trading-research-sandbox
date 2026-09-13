"""Execute compiled V1 conditions on causal research-event rows only."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import isfinite

from .ai_researcher import AIResearchPacket, _sha, _digest
from .hypothesis_compiler import CompiledHypothesis
from sandbox.market_state.discovery_dataset import DiscoveryDataset
from sandbox.strategies.dsl.conditions import ComparisonCondition, AndCondition
from sandbox.strategies.dsl.execution import MarketEntry, FixedPipsStop, RiskMultipleTarget
from sandbox.execution.models import TradeIntent, EntryType, Direction

COMPILED_STRATEGY_EXECUTION_VERSION = "COMPILED_STRATEGY_EXECUTION_V1"
_METADATA_KEYS = {"execution_adapter", "hypothesis_id", "compiled_hypothesis_fingerprint",
    "strategy_spec_fingerprint", "research_packet_fingerprint", "rule_id", "source_row_index",
    "anchor_kind", "geometry_family_id", "geometry_level_id"}


def _nonblank(value):
    if type(value) is not str or not value.strip():
        raise ValueError("expected nonblank identifier")


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("expected finite numeric value excluding bool")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("value exceeds finite float range") from exc
    if not isfinite(value):
        raise ValueError("numeric value must be finite")
    return value


def _timestamp(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("anchor timestamp must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


def _value(value, kind):
    if value is None:
        return None
    if kind == "NUMERIC":
        return _number(value)
    if type(value) not in (bool, int, float, str) or (type(value) is float and not isfinite(value)):
        raise ValueError("unsupported categorical value")
    return value


def _intent_id(compiled_fingerprint, dataset_id, row_index, timestamp, rule_id):
    return "DSL-" + _digest(dict(version=COMPILED_STRATEGY_EXECUTION_VERSION,
        compiled_hypothesis_fingerprint=compiled_fingerprint, dataset_id=dataset_id,
        source_row_index=row_index, anchor_evidence_timestamp=timestamp.isoformat(), rule_id=rule_id))[:24]


def _payload(batch):
    payload = {name: getattr(batch, name) for name in batch.__dataclass_fields__ if name != "fingerprint"}
    payload["intents"] = [intent.model_dump(mode="json") for intent in batch.intents]
    return payload


def _geometry(reference, stop, target, direction):
    if any(_number(v) <= 0 for v in (reference, stop, target)):
        raise ValueError("entry and exit prices must be positive")
    if direction == Direction.LONG:
        valid = stop < reference < target
    elif direction == Direction.SHORT:
        valid = target < reference < stop
    else:
        valid = False
    if not valid:
        raise ValueError("invalid entry/stop/target geometry")


@dataclass(frozen=True)
class CompiledStrategyIntentBatch:
    version: str
    compiled_hypothesis_fingerprint: str
    strategy_spec_fingerprint: str
    research_packet_fingerprint: str
    hypothesis_id: str
    strategy_id: str
    dataset_id: str
    symbol: str
    pip_size: float
    source_row_count: int
    matched_row_count: int
    intent_count: int
    matched_source_row_indices: tuple[int, ...]
    intents: tuple[TradeIntent, ...]
    fingerprint: str

    def __post_init__(self):
        if self.version != COMPILED_STRATEGY_EXECUTION_VERSION:
            raise ValueError("unsupported execution version")
        for value in (self.compiled_hypothesis_fingerprint, self.strategy_spec_fingerprint,
                      self.research_packet_fingerprint, self.fingerprint):
            _sha(value)
        for value in (self.hypothesis_id, self.strategy_id, self.dataset_id, self.symbol):
            _nonblank(value)
        if type(self.pip_size) is not float or _number(self.pip_size) <= 0:
            raise ValueError("pip size must be positive finite float")
        if any(type(c) is not int or c < 0 for c in (self.source_row_count, self.matched_row_count, self.intent_count)):
            raise ValueError("counts must be nonnegative exact integers")
        if self.matched_row_count > self.source_row_count:
            raise ValueError("matched count exceeds source count")
        indices = self.matched_source_row_indices
        if type(indices) is not tuple or len(indices) != self.matched_row_count or any(type(i) is not int or not 0 <= i < self.source_row_count for i in indices):
            raise ValueError("invalid matched row indices")
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError("matched indices must be strictly increasing")
        if type(self.intents) is not tuple or len(self.intents) != self.intent_count or any(type(i) is not TradeIntent for i in self.intents):
            raise ValueError("invalid intent tuple")
        row_order = []
        identifiers = set()
        for intent in self.intents:
            TradeIntent.model_validate(intent.model_dump(mode="python"))
            if (intent.strategy_id != self.strategy_id or intent.experiment_id != self.hypothesis_id or
                intent.dataset_id != self.dataset_id or intent.symbol != self.symbol or intent.requested_entry_type != EntryType.MARKET_CLOSE):
                raise ValueError("intent execution provenance mismatch")
            _timestamp(intent.signal_timestamp)
            _geometry(intent.requested_entry_price, intent.stop_loss, intent.take_profit, intent.direction)
            metadata = intent.metadata
            if set(metadata) != _METADATA_KEYS:
                raise ValueError("intent metadata must contain only execution provenance")
            expected = dict(execution_adapter="COMPILED_DISCOVERY_DSL_V1", hypothesis_id=self.hypothesis_id,
                compiled_hypothesis_fingerprint=self.compiled_hypothesis_fingerprint,
                strategy_spec_fingerprint=self.strategy_spec_fingerprint, research_packet_fingerprint=self.research_packet_fingerprint)
            if any(metadata[k] != v for k, v in expected.items()):
                raise ValueError("intent fingerprint provenance mismatch")
            row_index = metadata["source_row_index"]
            if type(row_index) is not int or row_index not in indices:
                raise ValueError("intent source index must be a matched row")
            for key in ("rule_id", "anchor_kind", "geometry_family_id", "geometry_level_id"):
                _nonblank(metadata[key])
            if intent.trade_intent_id != _intent_id(self.compiled_hypothesis_fingerprint, self.dataset_id,
                row_index, intent.signal_timestamp, metadata["rule_id"]) or intent.trade_intent_id in identifiers:
                raise ValueError("invalid or duplicate deterministic intent ID")
            identifiers.add(intent.trade_intent_id)
            row_order.append(row_index)
        if tuple(sorted(set(row_order))) != indices or any(a > b for a, b in zip(row_order, row_order[1:])):
            raise ValueError("intent rows must match batch membership and order")
        if self.fingerprint != _digest(_payload(self)):
            raise ValueError("intent batch fingerprint mismatch")


def _check_condition(condition, kinds, predictor_ids, referenced):
    if type(condition) is AndCondition:
        for child in condition.children:
            _check_condition(child, kinds, predictor_ids, referenced)
        return
    if type(condition) is not ComparisonCondition:
        raise ValueError("V1 supports only comparison and AND conditions")
    ref = condition.reference
    field = ref.key
    if ref.namespace != "feature" or ref.timeframe is not None or ref.context is not None:
        raise ValueError("V1 requires exact discovery feature references")
    if not field.startswith("x.") or field not in kinds or field not in predictor_ids:
        raise ValueError("condition feature must exist in dataset and packet")
    kind = kinds[field]
    if condition.operator.value in ("IN", "NOT_IN"):
        if type(condition.value) not in (list, tuple) or not condition.value:
            raise ValueError("membership condition requires nonempty scalar sequence")
        for value in condition.value:
            _value(value, kind)
    else:
        _value(condition.value, kind)
    if condition.operator.value in ("GT", "GTE", "LT", "LTE") and (kind != "NUMERIC" or condition.value is None):
        raise ValueError("ordered comparisons require numeric operands")
    referenced.add(field)


def _evaluate(condition, values, kinds):
    if type(condition) is AndCondition:
        return all(_evaluate(c, values, kinds) for c in condition.children)
    kind = kinds[condition.reference.key]
    left = values[condition.reference.key]
    op = condition.operator.value
    def equal(right):
        right = _value(right, kind)
        return left == right if kind == "NUMERIC" else type(left) is type(right) and left == right
    if op in ("IN", "NOT_IN"):
        matched = any(equal(v) for v in condition.value)
        return matched if op == "IN" else not matched
    if op in ("EQ", "NE"):
        matched = equal(condition.value)
        return matched if op == "EQ" else not matched
    if left is None:
        return False
    right = _number(condition.value)
    if op == "GT": return left > right
    if op == "GTE": return left >= right
    if op == "LT": return left < right
    if op == "LTE": return left <= right
    raise ValueError("unsupported comparison operator")


def build_compiled_strategy_intents(compiled: CompiledHypothesis, packet: AIResearchPacket,
                                    dataset: DiscoveryDataset, *, dataset_id: str, symbol: str,
                                    pip_size: float) -> CompiledStrategyIntentBatch:
    if type(compiled) is not CompiledHypothesis or type(packet) is not AIResearchPacket or type(dataset) is not DiscoveryDataset:
        raise TypeError("expected exact compiled hypothesis, packet, and dataset types")
    compiled = replace(compiled)
    packet.__post_init__()
    if compiled.research_packet_fingerprint != packet.fingerprint or compiled.target_field_id != packet.target_field_id:
        raise ValueError("compiled hypothesis must match packet")
    _nonblank(dataset_id); _nonblank(symbol)
    pip_size = _number(pip_size)
    if pip_size <= 0:
        raise ValueError("pip size must be positive")
    kinds = dict(zip(packet.feature_ids, packet.feature_kinds))
    kinds = {k: v for k, v in kinds.items() if k in packet.allowed_condition_feature_ids}
    referenced = set()
    for rule in compiled.strategy_spec.rules:
        _check_condition(rule.condition, kinds, dataset.predictor_field_ids, referenced)
        if type(rule.entry) is not MarketEntry or type(rule.stop) is not FixedPipsStop or type(rule.target) is not RiskMultipleTarget:
            raise ValueError("V1 requires market entry, fixed-pips stop, and risk-multiple target")
    if len(set(dataset.predictor_field_ids)) != len(dataset.predictor_field_ids):
        raise ValueError("duplicate predictor ID")
    predictor_index = {field: dataset.predictor_field_ids.index(field) for field in referenced}

    intents, matched = [], []
    for row_index, row in enumerate(dataset.rows):
        values = {
            field: _value(row.predictors[index].value, kinds[field])
            for field, index in predictor_index.items()
        }
        matched_rules = [rule for rule in compiled.strategy_spec.rules
                         if _evaluate(rule.condition, values, kinds)]
        if not matched_rules:
            continue
        anchor = row.record.anchor
        timestamp = _timestamp(anchor.evidence_end_utc)
        reference = _number(anchor.outcome_anchor.reference_price)
        if reference <= 0:
            raise ValueError("anchor reference price must be positive")
        for rule in matched_rules:
            risk = rule.stop.pips * pip_size
            direction = Direction(rule.entry.side.value)
            sign = 1 if direction == Direction.LONG else -1
            stop = reference - sign * risk
            target = reference + sign * risk * rule.target.multiple
            _geometry(reference, stop, target, direction)
            metadata = dict(execution_adapter="COMPILED_DISCOVERY_DSL_V1", hypothesis_id=compiled.hypothesis_id,
                compiled_hypothesis_fingerprint=compiled.fingerprint, strategy_spec_fingerprint=compiled.strategy_spec_fingerprint,
                research_packet_fingerprint=packet.fingerprint, rule_id=rule.rule_id, source_row_index=row_index,
                anchor_kind=anchor.kind.value, geometry_family_id=anchor.geometry_family_id, geometry_level_id=anchor.level_id)
            intents.append(TradeIntent(trade_intent_id=_intent_id(compiled.fingerprint, dataset_id, row_index, timestamp, rule.rule_id),
                strategy_id=compiled.strategy_spec.strategy_id, experiment_id=compiled.hypothesis_id, dataset_id=dataset_id,
                symbol=symbol, direction=direction, signal_timestamp=timestamp, requested_entry_type=EntryType.MARKET_CLOSE,
                requested_entry_price=reference, stop_loss=stop, take_profit=target, metadata=metadata))
            if not matched or matched[-1] != row_index:
                matched.append(row_index)
    state = dict(version=COMPILED_STRATEGY_EXECUTION_VERSION, compiled_hypothesis_fingerprint=compiled.fingerprint,
        strategy_spec_fingerprint=compiled.strategy_spec_fingerprint, research_packet_fingerprint=packet.fingerprint,
        hypothesis_id=compiled.hypothesis_id, strategy_id=compiled.strategy_spec.strategy_id, dataset_id=dataset_id,
        symbol=symbol, pip_size=pip_size, source_row_count=len(dataset.rows), matched_row_count=len(matched),
        intent_count=len(intents), matched_source_row_indices=tuple(matched), intents=tuple(intents))
    payload = {**state, "intents": [intent.model_dump(mode="json") for intent in intents]}
    return CompiledStrategyIntentBatch(**state, fingerprint=_digest(payload))
