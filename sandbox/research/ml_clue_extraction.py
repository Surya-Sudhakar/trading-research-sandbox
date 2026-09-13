"""Deterministic model-native feature evidence with artifact provenance."""
from dataclasses import dataclass, asdict
import hashlib
import json
from math import isfinite, isclose
from operator import index

import lightgbm

from .ml_lightgbm import LightGBMClueModel
from .ml_evaluation import MLPredictionEvaluation

ML_CLUE_REPORT_VERSION = "ML_CLUE_REPORT_V1"


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")


def _digest(state):
    payload = dict(state)
    payload["feature_clues"] = [asdict(c) for c in state["feature_clues"]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _float(value):
    if type(value) is not float or not isfinite(value) or value < 0:
        raise ValueError("evidence must be a nonnegative finite float")
    return 0.0 if value == 0 else value


def _orders(clues):
    return (tuple(c.field_id for c in sorted(clues, key=lambda c: (-c.gain, -c.split_count, c.field_id))),
            tuple(c.field_id for c in sorted(clues, key=lambda c: (-c.split_count, -c.gain, c.field_id))))


@dataclass(frozen=True)
class MLFeatureClue:
    field_id: str
    feature_kind: str
    feature_index: int
    split_count: int
    gain: float
    split_fraction: float
    gain_fraction: float

    def __post_init__(self):
        if type(self.field_id) is not str or not self.field_id.startswith("x."):
            raise ValueError("feature field must be in x. namespace")
        if type(self.feature_kind) is not str or self.feature_kind not in ("NUMERIC", "CATEGORICAL"):
            raise ValueError("unsupported feature kind")
        for value in (self.feature_index, self.split_count):
            if type(value) is not int or value < 0:
                raise ValueError("indices and counts must be nonnegative exact integers")
        for name in ("gain", "split_fraction", "gain_fraction"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        if self.split_fraction > 1 or self.gain_fraction > 1:
            raise ValueError("fractions must be in [0, 1]")


@dataclass(frozen=True)
class MLClueReport:
    version: str
    model_fingerprint: str
    model_sha256: str
    training_matrix_fingerprint: str
    preprocessor_fingerprint: str
    evaluation_fingerprint: str
    target_field_id: str
    feature_count: int
    total_split_count: int
    total_gain: float
    feature_clues: tuple[MLFeatureClue, ...]
    gain_order: tuple[str, ...]
    split_order: tuple[str, ...]
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != ML_CLUE_REPORT_VERSION:
            raise ValueError("unsupported clue report version")
        for value in (self.model_fingerprint, self.model_sha256, self.training_matrix_fingerprint,
                      self.preprocessor_fingerprint, self.evaluation_fingerprint, self.fingerprint):
            _sha(value)
        if type(self.target_field_id) is not str or not self.target_field_id.startswith("y."):
            raise ValueError("target must be in y. namespace")
        if type(self.feature_count) is not int or self.feature_count < 1:
            raise ValueError("feature count must be positive integer")
        if type(self.total_split_count) is not int or self.total_split_count < 0:
            raise ValueError("total splits must be nonnegative integer")
        object.__setattr__(self, "total_gain", _float(self.total_gain))
        clues = self.feature_clues
        if not isinstance(clues, tuple) or len(clues) != self.feature_count or any(type(c) is not MLFeatureClue for c in clues):
            raise ValueError("feature clues must be a matching nonempty tuple")
        if tuple(c.feature_index for c in clues) != tuple(range(self.feature_count)):
            raise ValueError("feature indices must preserve model order")
        if len({c.field_id for c in clues}) != self.feature_count:
            raise ValueError("duplicate feature fields")
        if sum(c.split_count for c in clues) != self.total_split_count or sum(c.gain for c in clues) != self.total_gain:
            raise ValueError("totals do not match feature evidence")
        for c in clues:
            if c.split_fraction != (c.split_count / self.total_split_count if self.total_split_count else 0.0):
                raise ValueError("split fraction does not match raw counts")
            if c.gain_fraction != (c.gain / self.total_gain if self.total_gain else 0.0):
                raise ValueError("gain fraction does not match raw gain")
        for total, fractions in ((self.total_gain, (c.gain_fraction for c in clues)),
                                 (self.total_split_count, (c.split_fraction for c in clues))):
            if not isclose(sum(fractions), 1.0 if total else 0.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("fraction sum is inconsistent")
        gain_order, split_order = _orders(clues)
        if not isinstance(self.gain_order, tuple) or self.gain_order != gain_order:
            raise ValueError("incorrect gain order")
        if not isinstance(self.split_order, tuple) or self.split_order != split_order:
            raise ValueError("incorrect split order")
        state = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "fingerprint"}
        if self.fingerprint != _digest(state):
            raise ValueError("clue report fingerprint mismatch")


def extract_ml_clues(model: LightGBMClueModel, evaluation: MLPredictionEvaluation) -> MLClueReport:
    if type(model) is not LightGBMClueModel or type(evaluation) is not MLPredictionEvaluation:
        raise TypeError("expected exact model and evaluation artifact types")
    if evaluation.model_fingerprint != model.fingerprint or evaluation.target_field_id != model.target_field_id:
        raise ValueError("evaluation must match model and target")
    booster = lightgbm.Booster(model_str=model.model_text)
    count = len(model.feature_specs)
    if booster.num_feature() != count:
        raise ValueError("booster feature count does not match model schema")
    if hasattr(booster, "feature_name") and tuple(booster.feature_name()) != tuple(s.field_id for s in model.feature_specs):
        raise ValueError("booster feature names do not match model schema order")
    iteration = booster.current_iteration()
    raw_splits = tuple(booster.feature_importance(importance_type="split", iteration=iteration))
    raw_gains = tuple(booster.feature_importance(importance_type="gain", iteration=iteration))
    if len(raw_splits) != count or len(raw_gains) != count:
        raise ValueError("importance lengths do not match feature count")
    splits = []
    for value in raw_splits:
        if isinstance(value, bool):
            raise ValueError("split counts cannot be bool")
        try:
            value = int(index(value))
        except TypeError as exc:
            raise ValueError("split counts must be integers") from exc
        if value < 0:
            raise ValueError("split counts cannot be negative")
        splits.append(value)
    try:
        gains = tuple(_float(float(v)) for v in raw_gains)
    except (TypeError, OverflowError) as exc:
        raise ValueError("gains must be finite numeric values") from exc
    total_splits, total_gain = sum(splits), _float(sum(gains))
    clues = tuple(MLFeatureClue(spec.field_id, spec.feature_kind, i, splits[i], gains[i],
        splits[i] / total_splits if total_splits else 0.0, gains[i] / total_gain if total_gain else 0.0)
        for i, spec in enumerate(model.feature_specs))
    gain_order, split_order = _orders(clues)
    state = dict(version=ML_CLUE_REPORT_VERSION, model_fingerprint=model.fingerprint,
        model_sha256=model.model_sha256, training_matrix_fingerprint=model.training_matrix_fingerprint,
        preprocessor_fingerprint=model.preprocessor_fingerprint, evaluation_fingerprint=evaluation.fingerprint,
        target_field_id=model.target_field_id, feature_count=count, total_split_count=total_splits,
        total_gain=total_gain, feature_clues=clues, gain_order=gain_order, split_order=split_order)
    return MLClueReport(**state, fingerprint=_digest(state))
