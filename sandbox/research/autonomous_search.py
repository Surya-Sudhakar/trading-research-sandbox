from __future__ import annotations

"""Pure deterministic candidate generation for Discovery research search spaces."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import combinations, product
import json
import math
from pathlib import Path
import re
from typing import Any

from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.research.research_memory import ResearchMemory


AUTONOMOUS_SEARCH_VERSION = "AUTONOMOUS_SEARCH_V1"


class SearchFieldKind(StrEnum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class NumericOperator(StrEnum):
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


class RejectionReason(StrEnum):
    REJECT_LOW_SAMPLE = "REJECT_LOW_SAMPLE"
    REJECT_NO_EFFECT = "REJECT_NO_EFFECT"
    REJECT_NEGATIVE_EFFECT = "REJECT_NEGATIVE_EFFECT"
    REJECT_UNSTABLE_ACROSS_YEARS = "REJECT_UNSTABLE_ACROSS_YEARS"
    REJECT_PARAMETER_SENSITIVE = "REJECT_PARAMETER_SENSITIVE"
    REJECT_SESSION_DEPENDENT = "REJECT_SESSION_DEPENDENT"
    REJECT_SPREAD_SENSITIVE = "REJECT_SPREAD_SENSITIVE"
    REJECT_OOS_FAILURE = "REJECT_OOS_FAILURE"
    REJECT_DUPLICATE_HYPOTHESIS = "REJECT_DUPLICATE_HYPOTHESIS"


REJECT_LOW_SAMPLE = RejectionReason.REJECT_LOW_SAMPLE.value
REJECT_NO_EFFECT = RejectionReason.REJECT_NO_EFFECT.value
REJECT_NEGATIVE_EFFECT = RejectionReason.REJECT_NEGATIVE_EFFECT.value
REJECT_UNSTABLE_ACROSS_YEARS = RejectionReason.REJECT_UNSTABLE_ACROSS_YEARS.value
REJECT_PARAMETER_SENSITIVE = RejectionReason.REJECT_PARAMETER_SENSITIVE.value
REJECT_SESSION_DEPENDENT = RejectionReason.REJECT_SESSION_DEPENDENT.value
REJECT_SPREAD_SENSITIVE = RejectionReason.REJECT_SPREAD_SENSITIVE.value
REJECT_OOS_FAILURE = RejectionReason.REJECT_OOS_FAILURE.value
REJECT_DUPLICATE_HYPOTHESIS = RejectionReason.REJECT_DUPLICATE_HYPOTHESIS.value


_FIELD_ID = re.compile(r"[xy]\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")
_NUMERIC_OPERATORS = tuple(NumericOperator)


def _identifier(name: str, value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _field_id(value: object, namespace: str) -> str:
    if not isinstance(value, str) or _FIELD_ID.fullmatch(value) is None or not value.startswith(namespace + "."):
        raise ValueError(f"field_id must be a valid {namespace}.* identifier")
    return value


def _scalar(value: object) -> bool | int | float | str:
    if type(value) not in (bool, int, float, str):
        raise ValueError("categorical values must be bool, int, finite float, or string")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("categorical values must be finite")
    if type(value) is str and not value:
        raise ValueError("categorical string values must be nonempty")
    return value


def _positive_values(name: str, values) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(type(value) is not int or value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


@dataclass(frozen=True, slots=True)
class SearchField:
    field_id: str
    kind: SearchFieldKind
    categorical_values: tuple[bool | int | float | str, ...] = ()
    threshold_values: tuple[float, ...] = ()
    operators: tuple[NumericOperator, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_id", _field_id(self.field_id, "x"))
        try:
            kind = SearchFieldKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("kind must be categorical or numeric") from exc
        object.__setattr__(self, "kind", kind)
        categorical = tuple(_scalar(value) for value in self.categorical_values)
        thresholds = tuple(self.threshold_values)
        operators = tuple(self.operators)
        if kind is SearchFieldKind.CATEGORICAL:
            if not categorical or thresholds or operators:
                raise ValueError("categorical field requires only categorical_values")
            keys = tuple(canonical_json({"value": value}) for value in categorical)
            if len(set(keys)) != len(keys):
                raise ValueError("categorical_values contains duplicates")
        else:
            if categorical or not thresholds:
                raise ValueError("numeric field requires only threshold_values")
            converted = []
            for value in thresholds:
                if type(value) not in (int, float) or type(value) is bool or not math.isfinite(float(value)):
                    raise ValueError("threshold_values must contain finite numbers excluding bool")
                converted.append(float(value))
            thresholds = tuple(converted)
            if len(set(thresholds)) != len(thresholds):
                raise ValueError("threshold_values contains duplicates")
            if not operators:
                operators = _NUMERIC_OPERATORS
            try:
                operators = tuple(NumericOperator(value) for value in operators)
            except (TypeError, ValueError) as exc:
                raise ValueError("unsupported numeric operator") from exc
            if len(set(operators)) != len(operators):
                raise ValueError("operators contains duplicates")
        object.__setattr__(self, "categorical_values", categorical)
        object.__setattr__(self, "threshold_values", thresholds)
        object.__setattr__(self, "operators", operators)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchField":
        if not isinstance(payload, dict):
            raise TypeError("search field must be a JSON object")
        allowed = {"field_id", "kind", "categorical_values", "threshold_values", "operators"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown search field keys: " + ", ".join(unknown))
        missing = sorted({"field_id", "kind"} - set(payload))
        if missing:
            raise ValueError("missing search field keys: " + ", ".join(missing))
        return cls(
            field_id=payload["field_id"], kind=payload["kind"],
            categorical_values=tuple(payload.get("categorical_values", ())),
            threshold_values=tuple(payload.get("threshold_values", ())),
            operators=tuple(payload.get("operators", ())),
        )


@dataclass(frozen=True, slots=True)
class SearchSpaceSpec:
    search_id: str
    experiment_id: str
    partition_id: str
    target_field_ids: tuple[str, ...]
    fields: tuple[SearchField, ...]
    swing_left_values: tuple[int, ...]
    swing_right_values: tuple[int, ...]
    swing_count_window_values: tuple[int, ...]
    max_conditions_per_candidate: int
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("search_id", "experiment_id", "partition_id"):
            object.__setattr__(self, name, _identifier(name, getattr(self, name)))
        object.__setattr__(self, "hypothesis_id", _identifier("hypothesis_id", self.hypothesis_id, optional=True))
        targets = tuple(_field_id(value, "y") for value in self.target_field_ids)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_field_ids must be nonempty and unique")
        fields = tuple(self.fields)
        if not fields or any(not isinstance(field, SearchField) for field in fields):
            raise ValueError("fields must contain at least one SearchField")
        if len({field.field_id for field in fields}) != len(fields):
            raise ValueError("fields contains duplicate field IDs")
        object.__setattr__(self, "target_field_ids", targets)
        object.__setattr__(self, "fields", fields)
        for name in ("swing_left_values", "swing_right_values", "swing_count_window_values"):
            object.__setattr__(self, name, _positive_values(name, getattr(self, name)))
        if type(self.max_conditions_per_candidate) is not int or self.max_conditions_per_candidate <= 0:
            raise ValueError("max_conditions_per_candidate must be a positive integer")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchSpaceSpec":
        if not isinstance(payload, dict):
            raise TypeError("search specification must be a JSON object")
        allowed = {
            "search_id", "experiment_id", "hypothesis_id", "partition_id",
            "target_field_ids", "fields", "swing_left_values", "swing_right_values",
            "swing_count_window_values", "max_conditions_per_candidate",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown search specification keys: " + ", ".join(unknown))
        required = allowed - {"hypothesis_id"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError("missing search specification keys: " + ", ".join(missing))
        return cls(
            search_id=payload["search_id"], experiment_id=payload["experiment_id"],
            hypothesis_id=payload.get("hypothesis_id"), partition_id=payload["partition_id"],
            target_field_ids=tuple(payload["target_field_ids"]),
            fields=tuple(SearchField.from_dict(field) for field in payload["fields"]),
            swing_left_values=tuple(payload["swing_left_values"]),
            swing_right_values=tuple(payload["swing_right_values"]),
            swing_count_window_values=tuple(payload["swing_count_window_values"]),
            max_conditions_per_candidate=payload["max_conditions_per_candidate"],
        )


@dataclass(frozen=True, slots=True)
class CandidateCondition:
    field_id: str
    operator: str
    value: bool | int | float | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_id", _field_id(self.field_id, "x"))
        if self.operator not in {"=", *(operator.value for operator in NumericOperator)}:
            raise ValueError("unsupported candidate condition operator")
        if self.operator == "=":
            object.__setattr__(self, "value", _scalar(self.value))
        elif type(self.value) not in (int, float) or type(self.value) is bool or not math.isfinite(float(self.value)):
            raise ValueError("numeric condition value must be finite and exclude bool")
        else:
            object.__setattr__(self, "value", float(self.value))


def _condition_key(condition: CandidateCondition) -> str:
    return canonical_json(asdict(condition))


def candidate_fingerprint_payload(
    *, search_id: str, experiment_id: str, partition_id: str,
    conditions, target_field_ids, swing_left_bars: int,
    swing_right_bars: int, swing_count_window: int,
) -> dict[str, Any]:
    materialized = tuple(conditions)
    if any(not isinstance(condition, CandidateCondition) for condition in materialized):
        raise TypeError("conditions must contain CandidateCondition")
    return {
        "version": AUTONOMOUS_SEARCH_VERSION,
        "search_id": search_id,
        "experiment_id": experiment_id,
        "partition_id": partition_id,
        "conditions": [asdict(condition) for condition in sorted(materialized, key=_condition_key)],
        "target_field_ids": sorted(target_field_ids),
        "swing_left_bars": swing_left_bars,
        "swing_right_bars": swing_right_bars,
        "swing_count_window": swing_count_window,
    }


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    candidate_fingerprint: str
    search_id: str
    experiment_id: str
    partition_id: str
    conditions: tuple[CandidateCondition, ...]
    target_field_ids: tuple[str, ...]
    swing_left_bars: int
    swing_right_bars: int
    swing_count_window: int
    hypothesis_id: str | None = None
    parent_candidate_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("search_id", "experiment_id", "partition_id"):
            _identifier(name, getattr(self, name))
        _identifier("hypothesis_id", self.hypothesis_id, optional=True)
        _identifier("parent_candidate_id", self.parent_candidate_id, optional=True)
        conditions = tuple(self.conditions)
        if not conditions or any(not isinstance(condition, CandidateCondition) for condition in conditions):
            raise ValueError("candidate conditions must be nonempty CandidateCondition values")
        keys = tuple(_condition_key(condition) for condition in conditions)
        if len(set(keys)) != len(keys):
            raise ValueError("candidate contains duplicate conditions")
        targets = tuple(_field_id(value, "y") for value in self.target_field_ids)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("candidate target_field_ids must be nonempty and unique")
        for name in ("swing_left_bars", "swing_right_bars", "swing_count_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "target_field_ids", targets)
        payload = candidate_fingerprint_payload(
            search_id=self.search_id, experiment_id=self.experiment_id,
            partition_id=self.partition_id, conditions=conditions,
            target_field_ids=targets, swing_left_bars=self.swing_left_bars,
            swing_right_bars=self.swing_right_bars,
            swing_count_window=self.swing_count_window,
        )
        expected = sha256_canonical(payload)
        if self.candidate_fingerprint != expected:
            raise ValueError("candidate_fingerprint does not match scientific definition")
        if self.candidate_id != "CAND-" + expected[:20].upper():
            raise ValueError("candidate_id does not match candidate fingerprint")


def load_search_space_spec(path: Path) -> SearchSpaceSpec:
    return SearchSpaceSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _atomic_conditions(spec: SearchSpaceSpec) -> tuple[CandidateCondition, ...]:
    result = []
    for field in spec.fields:
        if field.kind is SearchFieldKind.CATEGORICAL:
            result.extend(CandidateCondition(field.field_id, "=", value)
                          for value in field.categorical_values)
        else:
            result.extend(CandidateCondition(field.field_id, operator.value, threshold)
                          for operator in field.operators for threshold in field.threshold_values)
    keys = tuple(_condition_key(condition) for condition in result)
    if len(set(keys)) != len(keys):
        raise ValueError("search space produces duplicate conditions")
    return tuple(result)


def generate_candidates(spec: SearchSpaceSpec) -> tuple[CandidateSpec, ...]:
    if not isinstance(spec, SearchSpaceSpec):
        raise TypeError("spec must be SearchSpaceSpec")
    atoms = _atomic_conditions(spec)
    candidates = []
    fingerprints = set()
    for condition_count in range(1, min(spec.max_conditions_per_candidate, len(atoms)) + 1):
        for selected in combinations(atoms, condition_count):
            for left, right, count_window in product(
                spec.swing_left_values, spec.swing_right_values,
                spec.swing_count_window_values,
            ):
                payload = candidate_fingerprint_payload(
                    search_id=spec.search_id, experiment_id=spec.experiment_id,
                    partition_id=spec.partition_id, conditions=selected,
                    target_field_ids=spec.target_field_ids, swing_left_bars=left,
                    swing_right_bars=right, swing_count_window=count_window,
                )
                fingerprint = sha256_canonical(payload)
                if fingerprint in fingerprints:
                    raise ValueError("search space generated a duplicate candidate")
                fingerprints.add(fingerprint)
                candidates.append(CandidateSpec(
                    candidate_id="CAND-" + fingerprint[:20].upper(),
                    candidate_fingerprint=fingerprint, search_id=spec.search_id,
                    experiment_id=spec.experiment_id, hypothesis_id=spec.hypothesis_id,
                    partition_id=spec.partition_id, conditions=selected,
                    target_field_ids=spec.target_field_ids, swing_left_bars=left,
                    swing_right_bars=right, swing_count_window=count_window,
                ))
    return tuple(candidates)


def filter_unseen_candidates(candidates, memory: ResearchMemory) -> tuple[CandidateSpec, ...]:
    materialized = tuple(candidates)
    if any(not isinstance(candidate, CandidateSpec) for candidate in materialized):
        raise TypeError("candidates must contain CandidateSpec")
    if not isinstance(memory, ResearchMemory):
        raise TypeError("memory must be ResearchMemory")
    return tuple(candidate for candidate in materialized
                 if not memory.candidate_exists(candidate.candidate_fingerprint))


__all__ = [
    "AUTONOMOUS_SEARCH_VERSION", "SearchFieldKind", "NumericOperator",
    "SearchField", "SearchSpaceSpec", "CandidateCondition", "CandidateSpec",
    "RejectionReason", "candidate_fingerprint_payload", "generate_candidates",
    "filter_unseen_candidates", "load_search_space_spec",
    "REJECT_LOW_SAMPLE", "REJECT_NO_EFFECT", "REJECT_NEGATIVE_EFFECT",
    "REJECT_UNSTABLE_ACROSS_YEARS", "REJECT_PARAMETER_SENSITIVE",
    "REJECT_SESSION_DEPENDENT", "REJECT_SPREAD_SENSITIVE", "REJECT_OOS_FAILURE",
    "REJECT_DUPLICATE_HYPOTHESIS",
]
