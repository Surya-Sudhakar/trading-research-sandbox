"""Compile untested structured Discovery hypotheses into the existing DSL."""
from dataclasses import dataclass
from sandbox.strategies.dsl.conditions import ValueReference, ComparisonCondition, AndCondition, ComparisonOperator
from sandbox.strategies.dsl.execution import MarketEntry, FixedPipsStop, RiskMultipleTarget
from sandbox.strategies.dsl.specification import TradeRule, StrategySpec
from .ai_researcher import (AIResearchPacket, HypothesisProposal, parse_hypothesis_proposal,
                           _sha, _digest, _machine, _namespace)

COMPILED_HYPOTHESIS_VERSION = "COMPILED_HYPOTHESIS_V1"


def _compiled_payload(artifact):
    return {name: artifact.strategy_spec.model_dump(mode="json") if name == "strategy_spec" else getattr(artifact, name)
            for name in artifact.__dataclass_fields__ if name != "fingerprint"}


@dataclass(frozen=True)
class CompiledHypothesis:
    version: str
    hypothesis_fingerprint: str
    research_packet_fingerprint: str
    hypothesis_id: str
    target_field_id: str
    strategy_spec: StrategySpec
    strategy_spec_fingerprint: str
    fingerprint: str

    def __post_init__(self):
        if self.version != COMPILED_HYPOTHESIS_VERSION:
            raise ValueError("unsupported compiled hypothesis version")
        for value in (self.hypothesis_fingerprint, self.research_packet_fingerprint, self.strategy_spec_fingerprint, self.fingerprint):
            _sha(value)
        _machine(self.hypothesis_id)
        _namespace(self.target_field_id, "y.")
        if type(self.strategy_spec) is not StrategySpec:
            raise TypeError("expected StrategySpec")
        validated = StrategySpec.model_validate(self.strategy_spec)
        object.__setattr__(self, "strategy_spec", validated)
        if self.strategy_spec_fingerprint != _digest(validated.model_dump(mode="json")):
            raise ValueError("strategy spec fingerprint mismatch")
        if self.fingerprint != _digest(_compiled_payload(self)):
            raise ValueError("compiled hypothesis fingerprint mismatch")


def compile_hypothesis(proposal: HypothesisProposal, packet: AIResearchPacket) -> CompiledHypothesis:
    if type(proposal) is not HypothesisProposal or type(packet) is not AIResearchPacket:
        raise TypeError("expected exact proposal and packet types")
    proposal = parse_hypothesis_proposal(proposal, packet)
    rules = []
    for rule in proposal.rules:
        conditions = tuple(ComparisonCondition(reference=ValueReference(namespace="feature", key=c.field_id),
            operator=ComparisonOperator(c.operator), value=list(c.value) if type(c.value) is tuple else c.value)
            for c in rule.conditions)
        condition = conditions[0] if len(conditions) == 1 else AndCondition(children=conditions)
        rules.append(TradeRule(rule_id=rule.rule_id, condition=condition, entry=MarketEntry(side=rule.side),
            stop=FixedPipsStop(pips=rule.stop_pips), target=RiskMultipleTarget(multiple=rule.target_r_multiple)))
    spec = StrategySpec(family_id=proposal.family_id, strategy_id=proposal.strategy_id,
                        variant_id=proposal.variant_id, rules=tuple(rules))
    payload = dict(version=COMPILED_HYPOTHESIS_VERSION, hypothesis_fingerprint=proposal.fingerprint,
        research_packet_fingerprint=packet.fingerprint, hypothesis_id=proposal.hypothesis_id,
        target_field_id=proposal.target_field_id, strategy_spec=spec.model_dump(mode="json"),
        strategy_spec_fingerprint=_digest(spec.model_dump(mode="json")))
    return CompiledHypothesis(**{**payload, "strategy_spec": spec}, fingerprint=_digest(payload))
