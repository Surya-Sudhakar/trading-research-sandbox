"""Deterministic regression clue artifacts; no evaluation or target labels."""
from dataclasses import dataclass, asdict
import hashlib
from math import isfinite

import lightgbm
import numpy as np

from .ml_discovery_matrix import MLFeatureSpec
from .ml_preprocessing import EncodedMLMatrix, _digest, _sha, _indices, _schema, _schema_payload, _state

MODEL_VERSION = "LIGHTGBM_CLUE_MODEL_V1"
PREDICTION_VERSION = "LIGHTGBM_PREDICTION_BATCH_V1"


@dataclass(frozen=True)
class LightGBMClueConfig:
    num_boost_round: int = 64
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 10
    seed: int = 1729

    def __post_init__(self):
        for value, minimum in ((self.num_boost_round, 1), (self.num_leaves, 2), (self.min_data_in_leaf, 1), (self.seed, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError("invalid integer model configuration")
        if type(self.learning_rate) is not float or not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be a positive finite float")


def _model_digest(state):
    payload = {key: value for key, value in state.items() if key != "model_text"}
    payload["feature_specs"] = _schema_payload(state["feature_specs"])
    payload["config"] = asdict(state["config"])
    return _digest(payload)


@dataclass(frozen=True)
class LightGBMClueModel:
    version: str
    lightgbm_version: str
    training_matrix_fingerprint: str
    preprocessor_fingerprint: str
    target_field_id: str
    feature_specs: tuple[MLFeatureSpec, ...]
    categorical_feature_indices: tuple[int, ...]
    training_row_indices: tuple[int, ...]
    config: LightGBMClueConfig
    model_text: str
    model_sha256: str
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != MODEL_VERSION:
            raise ValueError("unsupported model version")
        if not isinstance(self.lightgbm_version, str) or not self.lightgbm_version.strip():
            raise ValueError("LightGBM version is required")
        for value in (self.training_matrix_fingerprint, self.preprocessor_fingerprint, self.model_sha256, self.fingerprint):
            _sha(value)
        _schema(self.feature_specs)
        _indices(self.categorical_feature_indices, nonempty=False)
        if self.categorical_feature_indices != tuple(i for i, s in enumerate(self.feature_specs) if s.feature_kind == "CATEGORICAL"):
            raise ValueError("categorical indices must match features")
        _indices(self.training_row_indices)
        if not isinstance(self.target_field_id, str) or not self.target_field_id.startswith("y."):
            raise ValueError("target must be in y. namespace")
        if not isinstance(self.config, LightGBMClueConfig):
            raise ValueError("invalid model configuration")
        self.config.__post_init__()
        if not isinstance(self.model_text, str) or not self.model_text.strip():
            raise ValueError("model text is required")
        if hashlib.sha256(self.model_text.encode("utf-8")).hexdigest() != self.model_sha256:
            raise ValueError("model text hash mismatch")
        if self.fingerprint != _model_digest(_state(self)):
            raise ValueError("model fingerprint mismatch")


@dataclass(frozen=True)
class LightGBMPredictionBatch:
    version: str
    model_fingerprint: str
    encoded_matrix_fingerprint: str
    source_row_indices: tuple[int, ...]
    target_field_id: str
    predictions: tuple[float, ...]
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != PREDICTION_VERSION:
            raise ValueError("unsupported prediction version")
        for value in (self.model_fingerprint, self.encoded_matrix_fingerprint, self.fingerprint):
            _sha(value)
        _indices(self.source_row_indices)
        if not isinstance(self.target_field_id, str) or not self.target_field_id.startswith("y."):
            raise ValueError("target must be in y. namespace")
        if not isinstance(self.predictions, tuple) or len(self.predictions) != len(self.source_row_indices) or any(type(v) is not float or not isfinite(v) for v in self.predictions):
            raise ValueError("predictions must be finite floats with matching row count")
        if self.fingerprint != _digest(_state(self)):
            raise ValueError("prediction fingerprint mismatch")


def _x_array(rows):
    return np.asarray([[np.nan if v is None else v for v in row] for row in rows], dtype=np.float64)


def train_lightgbm_clue_model(matrix: EncodedMLMatrix,
                              config: LightGBMClueConfig = LightGBMClueConfig()) -> LightGBMClueModel:
    if not isinstance(matrix, EncodedMLMatrix) or not isinstance(config, LightGBMClueConfig):
        raise TypeError("expected EncodedMLMatrix and LightGBMClueConfig")
    positions = tuple(i for i, observed in enumerate(matrix.target_observed) if observed)
    if len(positions) < 2:
        raise ValueError("training requires at least two observed target rows")
    x = _x_array(tuple(matrix.x_rows[i] for i in positions))
    y = np.asarray([matrix.y_values[i] for i in positions], dtype=np.float64)
    params = dict(objective="regression", metric="l2", learning_rate=config.learning_rate,
        num_leaves=config.num_leaves, min_data_in_leaf=config.min_data_in_leaf,
        verbosity=-1, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
        seed=config.seed, feature_fraction_seed=config.seed, bagging_seed=config.seed,
        data_random_seed=config.seed, deterministic=True, force_col_wise=True,
        num_threads=1, feature_pre_filter=False)
    training = lightgbm.Dataset(x, label=y, feature_name=[s.field_id for s in matrix.feature_specs],
        categorical_feature=list(matrix.categorical_feature_indices), free_raw_data=False)
    booster = lightgbm.train(params, training, num_boost_round=config.num_boost_round)
    text = booster.model_to_string()
    state = dict(version=MODEL_VERSION, lightgbm_version=lightgbm.__version__,
        training_matrix_fingerprint=matrix.fingerprint, preprocessor_fingerprint=matrix.preprocessor_fingerprint,
        target_field_id=matrix.target_field_id, feature_specs=matrix.feature_specs,
        categorical_feature_indices=matrix.categorical_feature_indices,
        training_row_indices=tuple(matrix.source_row_indices[i] for i in positions), config=config,
        model_text=text, model_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return LightGBMClueModel(**state, fingerprint=_model_digest(state))


def predict_lightgbm_clue_model(model: LightGBMClueModel, matrix: EncodedMLMatrix) -> LightGBMPredictionBatch:
    if not isinstance(model, LightGBMClueModel) or not isinstance(matrix, EncodedMLMatrix):
        raise TypeError("expected LightGBMClueModel and EncodedMLMatrix")
    if (model.feature_specs != matrix.feature_specs or
        model.categorical_feature_indices != matrix.categorical_feature_indices or
        model.preprocessor_fingerprint != matrix.preprocessor_fingerprint or
        model.target_field_id != matrix.target_field_id):
        raise ValueError("prediction matrix must match model schema, target, and frozen preprocessor")
    booster = lightgbm.Booster(model_str=model.model_text)
    values = booster.predict(_x_array(matrix.x_rows), num_iteration=booster.current_iteration(), num_threads=1)
    predictions = tuple(float(v) for v in values)
    state = dict(version=PREDICTION_VERSION, model_fingerprint=model.fingerprint,
        encoded_matrix_fingerprint=matrix.fingerprint, source_row_indices=matrix.source_row_indices,
        target_field_id=matrix.target_field_id, predictions=predictions)
    return LightGBMPredictionBatch(**state, fingerprint=_digest(state))
