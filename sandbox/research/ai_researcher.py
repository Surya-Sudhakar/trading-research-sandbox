"""Immutable evidence packets and structured, unverified research proposals."""
from dataclasses import dataclass, asdict, fields
import hashlib
import json
from math import isfinite
import re

from .ml_clue_extraction import MLClueReport
from .ml_evaluation import MLPredictionEvaluation

AI_RESEARCH_PACKET_VERSION = "AI_RESEARCH_PACKET_V1"
HYPOTHESIS_PROPOSAL_VERSION = "HYPOTHESIS_PROPOSAL_V1"
HYPOTHESIS_STATUS = "UNVERIFIED_DISCOVERY_HYPOTHESIS"
SUMMARY_FIELDS = ("source_row_count", "observed_target_count", "missing_target_count",
    "actual_mean", "actual_median", "prediction_mean", "prediction_median",
    "mean_absolute_error", "median_absolute_error", "mean_squared_error", "root_mean_squared_error",
    "r_squared", "pearson_correlation", "spearman_rank_correlation")
RESEARCHER_INSTRUCTIONS = (
    "Evidence is observational/model evidence, not causal proof. Propose exactly ONE falsifiable "
    "Discovery hypothesis. Do not claim profitability or validation. Use only allowed_condition_feature_ids. "
    "target_field_id is the researched outcome. Choose LONG or SHORT explicitly; direction has not been proved. "
    "Provide explicit conditions and categorical values. Numeric thresholds are hypotheses to test, not "
    "discovered facts. Use market entry, a fixed-pips stop, and a risk-multiple target. The proposal will "
    "be tested on Discovery data. Future Validation/Final information is unavailable and prohibited."
)


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _state(obj):
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.name != "fingerprint"}


def _machine(value):
    if type(value) is not str or re.fullmatch(r"[A-Z0-9_]+", value) is None:
        raise ValueError("expected uppercase machine identifier")


def _namespace(value, prefix):
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError("invalid field namespace")


def _feature_ids(values):
    if type(values) is not tuple or not values:
        raise ValueError("feature IDs must be nonempty tuple")
    for value in values:
        _namespace(value, "x.")
    if len(set(values)) != len(values):
        raise ValueError("duplicate feature IDs")


def _scalar(value):
    if value is not None and type(value) not in (bool, int, float, str):
        raise ValueError("condition value must be a JSON scalar")
    if type(value) is float and not isfinite(value):
        raise ValueError("condition value must be finite")


@dataclass(frozen=True)
class AIResearchPacket:
    version: str
    clue_report_fingerprint: str
    evaluation_fingerprint: str
    model_fingerprint: str
    target_field_id: str
    feature_ids: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    feature_gain_fractions: tuple[float, ...]
    feature_split_fractions: tuple[float, ...]
    gain_order: tuple[str, ...]
    split_order: tuple[str, ...]
    evaluation_summary: tuple[tuple[str, object], ...]
    allowed_condition_feature_ids: tuple[str, ...]
    permitted_entry_kinds: tuple[str, ...]
    permitted_stop_kinds: tuple[str, ...]
    permitted_target_kinds: tuple[str, ...]
    researcher_instructions: str
    fingerprint: str

    def __post_init__(self):
        if self.version != AI_RESEARCH_PACKET_VERSION:
            raise ValueError("unsupported packet version")
        for value in (self.clue_report_fingerprint, self.evaluation_fingerprint, self.model_fingerprint, self.fingerprint):
            _sha(value)
        _namespace(self.target_field_id, "y.")
        _feature_ids(self.feature_ids)
        if type(self.feature_kinds) is not tuple or len(self.feature_kinds) != len(self.feature_ids) or any(type(k) is not str or k not in ("NUMERIC", "CATEGORICAL") for k in self.feature_kinds):
            raise ValueError("feature kinds must match features")
        for fractions in (self.feature_gain_fractions, self.feature_split_fractions):
            if type(fractions) is not tuple or len(fractions) != len(self.feature_ids) or any(type(v) is not float or not isfinite(v) or not 0 <= v <= 1 for v in fractions):
                raise ValueError("invalid feature fractions")
        for order in (self.gain_order, self.split_order):
            if type(order) is not tuple or len(order) != len(self.feature_ids) or any(type(v) is not str for v in order) or set(order) != set(self.feature_ids):
                raise ValueError("feature orders must contain each feature exactly once")
        if type(self.allowed_condition_feature_ids) is not tuple or self.allowed_condition_feature_ids != self.feature_ids:
            raise ValueError("allowed features must match evidence features")
        summary = self.evaluation_summary
        if type(summary) is not tuple or any(type(p) is not tuple or len(p) != 2 for p in summary) or tuple(p[0] for p in summary) != SUMMARY_FIELDS:
            raise ValueError("evaluation summary must preserve the fixed metric order")
        for i, (name, value) in enumerate(summary):
            if i < 3:
                if type(value) is not int or value < (0 if i == 2 else 2):
                    raise ValueError("invalid summary count")
            elif value is None and name in SUMMARY_FIELDS[-3:]:
                continue
            elif type(value) is not float or not isfinite(value):
                raise ValueError("invalid summary metric")
        if summary[1][1] + summary[2][1] != summary[0][1]:
            raise ValueError("inconsistent summary counts")
        if self.permitted_entry_kinds != ("market",) or self.permitted_stop_kinds != ("fixed_pips",) or self.permitted_target_kinds != ("r_multiple",):
            raise ValueError("unsupported execution permissions")
        if self.researcher_instructions != RESEARCHER_INSTRUCTIONS:
            raise ValueError("researcher instructions must be the fixed V1 text")
        if self.fingerprint != _digest(_state(self)):
            raise ValueError("packet fingerprint mismatch")


def build_research_packet(clues: MLClueReport, evaluation: MLPredictionEvaluation) -> AIResearchPacket:
    if type(clues) is not MLClueReport or type(evaluation) is not MLPredictionEvaluation:
        raise TypeError("expected exact clue and evaluation types")
    if (clues.evaluation_fingerprint != evaluation.fingerprint or clues.model_fingerprint != evaluation.model_fingerprint or
        clues.target_field_id != evaluation.target_field_id):
        raise ValueError("clues and evaluation must match")
    feature_ids = tuple(c.field_id for c in clues.feature_clues)
    state = dict(version=AI_RESEARCH_PACKET_VERSION, clue_report_fingerprint=clues.fingerprint,
        evaluation_fingerprint=evaluation.fingerprint, model_fingerprint=clues.model_fingerprint,
        target_field_id=clues.target_field_id, feature_ids=feature_ids,
        feature_kinds=tuple(c.feature_kind for c in clues.feature_clues),
        feature_gain_fractions=tuple(c.gain_fraction for c in clues.feature_clues),
        feature_split_fractions=tuple(c.split_fraction for c in clues.feature_clues),
        gain_order=clues.gain_order, split_order=clues.split_order,
        evaluation_summary=tuple((name, getattr(evaluation, name)) for name in SUMMARY_FIELDS),
        allowed_condition_feature_ids=feature_ids, permitted_entry_kinds=("market",),
        permitted_stop_kinds=("fixed_pips",), permitted_target_kinds=("r_multiple",),
        researcher_instructions=RESEARCHER_INSTRUCTIONS)
    return AIResearchPacket(**state, fingerprint=_digest(state))


@dataclass(frozen=True)
class HypothesisCondition:
    field_id: str
    operator: str
    value: object

    def __post_init__(self):
        _namespace(self.field_id, "x.")
        if type(self.operator) is not str or self.operator not in ("EQ", "NE", "GT", "GTE", "LT", "LTE", "IN", "NOT_IN"):
            raise ValueError("unsupported comparison operator")
        if self.operator in ("IN", "NOT_IN"):
            if type(self.value) is not tuple or not self.value:
                raise ValueError("membership comparison requires nonempty tuple")
            for value in self.value:
                _scalar(value)
        else:
            _scalar(self.value)


@dataclass(frozen=True)
class HypothesisRule:
    rule_id: str
    conditions: tuple[HypothesisCondition, ...]
    condition_logic: str
    side: str
    stop_pips: float
    target_r_multiple: float

    def __post_init__(self):
        _machine(self.rule_id)
        if type(self.conditions) is not tuple or not self.conditions or any(type(c) is not HypothesisCondition for c in self.conditions):
            raise ValueError("rule requires a nonempty condition tuple")
        for condition in self.conditions:
            condition.__post_init__()
        if self.condition_logic != "ALL" or self.side not in ("LONG", "SHORT"):
            raise ValueError("V1 requires ALL logic and explicit LONG or SHORT")
        for value in (self.stop_pips, self.target_r_multiple):
            if type(value) is not float or not isfinite(value) or value <= 0:
                raise ValueError("execution parameters must be positive finite floats")


def _proposal_digest(state):
    payload = dict(state)
    payload["rules"] = []
    for rule in state["rules"]:
        data = asdict(rule)
        for condition in data["conditions"]:
            value = condition["value"]
            condition["value"] = (["tuple", [[type(v).__name__, v] for v in value]]
                if type(value) is tuple else [type(value).__name__, value])
        payload["rules"].append(data)
    return _digest(payload)


@dataclass(frozen=True)
class HypothesisProposal:
    version: str
    status: str
    research_packet_fingerprint: str
    hypothesis_id: str
    statement: str
    rationale: str
    target_field_id: str
    family_id: str
    strategy_id: str
    variant_id: str
    evidence_feature_ids: tuple[str, ...]
    rules: tuple[HypothesisRule, ...]
    fingerprint: str

    def __post_init__(self):
        if self.version != HYPOTHESIS_PROPOSAL_VERSION or self.status != HYPOTHESIS_STATUS:
            raise ValueError("proposal must remain an unverified Discovery hypothesis")
        _sha(self.research_packet_fingerprint)
        _sha(self.fingerprint)
        for value in (self.hypothesis_id, self.family_id, self.strategy_id, self.variant_id):
            _machine(value)
        for value in (self.statement, self.rationale):
            if type(value) is not str or not value.strip():
                raise ValueError("statement and rationale must be nonblank")
        _namespace(self.target_field_id, "y.")
        _feature_ids(self.evidence_feature_ids)
        if type(self.rules) is not tuple or not self.rules or any(type(r) is not HypothesisRule for r in self.rules):
            raise ValueError("proposal requires nonempty rule tuple")
        for rule in self.rules:
            rule.__post_init__()
        if len({r.rule_id for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate rule IDs")
        if any(c.field_id not in self.evidence_feature_ids for r in self.rules for c in r.conditions):
            raise ValueError("condition feature omitted from evidence IDs")
        if self.fingerprint != _proposal_digest(_state(self)):
            raise ValueError("proposal fingerprint mismatch")


def _mapping(data, cls, optional=()):
    if type(data) is not dict:
        raise TypeError("structured object must be dict")
    names = {f.name for f in fields(cls)}
    if set(data) - names or names - set(optional) - set(data):
        raise ValueError("unexpected or missing structured fields")
    return dict(data)


def _sequence(value):
    if type(value) not in (list, tuple):
        raise ValueError("expected structured array")
    return tuple(value)


def parse_hypothesis_proposal(data: object, packet: AIResearchPacket) -> HypothesisProposal:
    if type(packet) is not AIResearchPacket:
        raise TypeError("expected AIResearchPacket")
    packet.__post_init__()
    if type(data) is HypothesisProposal:
        data.__post_init__()
        proposal = data
    else:
        state = _mapping(data, HypothesisProposal, optional=("fingerprint",))
        has_fingerprint = "fingerprint" in state
        supplied = state.pop("fingerprint", None)
        state["evidence_feature_ids"] = _sequence(state["evidence_feature_ids"])
        rules = []
        for raw_rule in _sequence(state["rules"]):
            rule = _mapping(raw_rule, HypothesisRule)
            conditions = []
            for raw_condition in _sequence(rule["conditions"]):
                condition = _mapping(raw_condition, HypothesisCondition)
                if type(condition["value"]) is list:
                    condition["value"] = tuple(condition["value"])
                conditions.append(HypothesisCondition(**condition))
            rule["conditions"] = tuple(conditions)
            rules.append(HypothesisRule(**rule))
        state["rules"] = tuple(rules)
        proposal = HypothesisProposal(**state, fingerprint=supplied if has_fingerprint else _proposal_digest(state))
    if proposal.research_packet_fingerprint != packet.fingerprint or proposal.target_field_id != packet.target_field_id:
        raise ValueError("proposal must match packet fingerprint and target")
    if any(f not in packet.allowed_condition_feature_ids for f in proposal.evidence_feature_ids) or any(
            c.field_id not in packet.allowed_condition_feature_ids for r in proposal.rules for c in r.conditions):
        raise ValueError("proposal uses a feature outside packet permissions")
    return proposal
