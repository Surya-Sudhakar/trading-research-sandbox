"""Discovery-fitted typed vocabularies and immutable encoded matrices."""
from dataclasses import dataclass, fields
import hashlib
import json
from math import isfinite

from .ml_discovery_matrix import MLDiscoveryMatrix, MLFeatureSpec

PREPROCESSOR_VERSION = "ML_PREPROCESSOR_V1"
ENCODED_MATRIX_VERSION = "ENCODED_ML_MATRIX_V1"
MISSING_CATEGORY_CODE = 0
UNKNOWN_CATEGORY_CODE = 1
_TYPE_ORDER = {bool: 0, int: 1, float: 2, str: 3}


def _category_key(value):
    if type(value) not in _TYPE_ORDER or (type(value) is float and not isfinite(value)):
        raise ValueError("category must be bool, int, finite float, or str")
    return (_TYPE_ORDER[type(value)], 0.0 if type(value) is float and value == 0 else value)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256 fingerprint")


def _indices(values, nonempty=True):
    if not isinstance(values, tuple) or (nonempty and not values) or any(type(i) is not int or i < 0 for i in values):
        raise ValueError("expected tuple of nonnegative integer indices")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("indices must be strictly increasing")


def _schema(specs):
    if not isinstance(specs, tuple) or not specs or any(not isinstance(s, MLFeatureSpec) for s in specs):
        raise ValueError("expected nonempty feature-spec tuple")
    for spec in specs:
        spec.__post_init__()
    if len({s.field_id for s in specs}) != len(specs):
        raise ValueError("duplicate feature fields")


def _schema_payload(specs):
    return [{"field_id": s.field_id, "feature_kind": s.feature_kind} for s in specs]


def _state(obj):
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.name != "fingerprint"}


def _preprocessor_digest(state):
    payload = dict(state)
    payload["feature_specs"] = _schema_payload(state["feature_specs"])
    payload["categorical_encodings"] = [{"field_id": e.field_id,
        "categories": [[type(v).__name__, v] for v in e.categories]} for e in state["categorical_encodings"]]
    return _digest(payload)


def _encoded_digest(state):
    payload = dict(state)
    payload["feature_specs"] = _schema_payload(state["feature_specs"])
    return _digest(payload)


@dataclass(frozen=True)
class CategoricalFeatureEncoding:
    field_id: str
    categories: tuple[object, ...]

    def __post_init__(self):
        if not isinstance(self.field_id, str) or not self.field_id.startswith("x."):
            raise ValueError("categorical field must be in x. namespace")
        if not isinstance(self.categories, tuple):
            raise ValueError("categories must be a tuple")
        keys = tuple(_category_key(v) for v in self.categories)
        if any(a >= b for a, b in zip(keys, keys[1:])):
            raise ValueError("categories must be unique and canonically sorted")
        object.__setattr__(self, "categories", tuple(0.0 if type(v) is float and v == 0 else v for v in self.categories))

    def _lookup(self):
        return {_category_key(v): i + 2 for i, v in enumerate(self.categories)}


@dataclass(frozen=True)
class MLPreprocessor:
    version: str
    fitted_from_matrix_fingerprint: str
    feature_specs: tuple[MLFeatureSpec, ...]
    categorical_encodings: tuple[CategoricalFeatureEncoding, ...]
    missing_category_code: int
    unknown_category_code: int
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != PREPROCESSOR_VERSION:
            raise ValueError("unsupported preprocessor version")
        _sha(self.fitted_from_matrix_fingerprint)
        _schema(self.feature_specs)
        if not isinstance(self.categorical_encodings, tuple) or any(not isinstance(e, CategoricalFeatureEncoding) for e in self.categorical_encodings):
            raise ValueError("invalid categorical encodings")
        if tuple(e.field_id for e in self.categorical_encodings) != tuple(s.field_id for s in self.feature_specs if s.feature_kind == "CATEGORICAL"):
            raise ValueError("categorical encodings must match ordered categorical features")
        if type(self.missing_category_code) is not int or self.missing_category_code != MISSING_CATEGORY_CODE:
            raise ValueError("missing category code must be 0")
        if type(self.unknown_category_code) is not int or self.unknown_category_code != UNKNOWN_CATEGORY_CODE:
            raise ValueError("unknown category code must be 1")
        _sha(self.fingerprint)
        if self.fingerprint != _preprocessor_digest(_state(self)):
            raise ValueError("preprocessor fingerprint does not match content")


@dataclass(frozen=True)
class EncodedMLMatrix:
    version: str
    source_matrix_fingerprint: str
    preprocessor_fingerprint: str
    source_row_indices: tuple[int, ...]
    feature_specs: tuple[MLFeatureSpec, ...]
    categorical_feature_indices: tuple[int, ...]
    x_rows: tuple[tuple[object, ...], ...]
    target_field_id: str
    y_values: tuple[float | None, ...]
    target_observed: tuple[bool, ...]
    observed_target_row_indices: tuple[int, ...]
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != ENCODED_MATRIX_VERSION:
            raise ValueError("unsupported encoded matrix version")
        _sha(self.source_matrix_fingerprint)
        _sha(self.preprocessor_fingerprint)
        _indices(self.source_row_indices)
        _schema(self.feature_specs)
        _indices(self.categorical_feature_indices, nonempty=False)
        if self.categorical_feature_indices != tuple(i for i, s in enumerate(self.feature_specs) if s.feature_kind == "CATEGORICAL"):
            raise ValueError("categorical indices must match feature schema")
        size = len(self.source_row_indices)
        if not isinstance(self.x_rows, tuple) or len(self.x_rows) != size:
            raise ValueError("encoded row count mismatch")
        for row in self.x_rows:
            if not isinstance(row, tuple) or len(row) != len(self.feature_specs):
                raise ValueError("encoded row width mismatch")
            for value, spec in zip(row, self.feature_specs):
                if spec.feature_kind == "CATEGORICAL":
                    if type(value) is not int or value < 0:
                        raise ValueError("category codes must be nonnegative exact integers")
                elif value is not None and (type(value) is not float or not isfinite(value)):
                    raise ValueError("numeric cells must be finite floats or None")
        if not isinstance(self.target_field_id, str) or not self.target_field_id.startswith("y."):
            raise ValueError("target must be in y. namespace")
        if not isinstance(self.y_values, tuple) or len(self.y_values) != size or any(v is not None and (type(v) is not float or not isfinite(v)) for v in self.y_values):
            raise ValueError("targets must be finite floats or None with matching row count")
        if not isinstance(self.target_observed, tuple) or any(type(v) is not bool for v in self.target_observed) or self.target_observed != tuple(v is not None for v in self.y_values):
            raise ValueError("target mask must match target missingness")
        _indices(self.observed_target_row_indices, nonempty=False)
        if self.observed_target_row_indices != tuple(i for i, present in zip(self.source_row_indices, self.target_observed) if present):
            raise ValueError("observed indices must reference original rows")
        _sha(self.fingerprint)
        if self.fingerprint != _encoded_digest(_state(self)):
            raise ValueError("encoded matrix fingerprint does not match content")


def fit_ml_preprocessor(discovery_matrix: MLDiscoveryMatrix) -> MLPreprocessor:
    if not isinstance(discovery_matrix, MLDiscoveryMatrix):
        raise TypeError("expected MLDiscoveryMatrix")
    encodings = []
    for position, spec in enumerate(discovery_matrix.feature_specs):
        if spec.feature_kind == "CATEGORICAL":
            categories = {_category_key(row[position]): row[position] for row in discovery_matrix.x_rows if row[position] is not None}
            encodings.append(CategoricalFeatureEncoding(spec.field_id, tuple(categories[key] for key in sorted(categories))))
    state = dict(version=PREPROCESSOR_VERSION, fitted_from_matrix_fingerprint=discovery_matrix.fingerprint,
        feature_specs=discovery_matrix.feature_specs, categorical_encodings=tuple(encodings),
        missing_category_code=MISSING_CATEGORY_CODE, unknown_category_code=UNKNOWN_CATEGORY_CODE)
    return MLPreprocessor(**state, fingerprint=_preprocessor_digest(state))


def transform_ml_matrix(matrix: MLDiscoveryMatrix, preprocessor: MLPreprocessor) -> EncodedMLMatrix:
    if not isinstance(matrix, MLDiscoveryMatrix) or not isinstance(preprocessor, MLPreprocessor):
        raise TypeError("expected MLDiscoveryMatrix and MLPreprocessor")
    if matrix.feature_specs != preprocessor.feature_specs:
        raise ValueError("matrix schema must exactly match fitted schema")
    lookups = {e.field_id: e._lookup() for e in preprocessor.categorical_encodings}
    rows = tuple(tuple(value if spec.feature_kind == "NUMERIC" else
        MISSING_CATEGORY_CODE if value is None else lookups[spec.field_id].get(_category_key(value), UNKNOWN_CATEGORY_CODE)
        for spec, value in zip(matrix.feature_specs, row)) for row in matrix.x_rows)
    state = dict(version=ENCODED_MATRIX_VERSION, source_matrix_fingerprint=matrix.fingerprint,
        preprocessor_fingerprint=preprocessor.fingerprint, source_row_indices=matrix.source_row_indices,
        feature_specs=matrix.feature_specs,
        categorical_feature_indices=tuple(i for i, s in enumerate(matrix.feature_specs) if s.feature_kind == "CATEGORICAL"),
        x_rows=rows, target_field_id=matrix.target_field_id, y_values=matrix.y_values,
        target_observed=matrix.target_observed, observed_target_row_indices=matrix.observed_target_row_indices)
    return EncodedMLMatrix(**state, fingerprint=_encoded_digest(state))
