"""Explicit predictor semantics and a canonical, immutable ML input boundary."""
from dataclasses import dataclass
import hashlib
import json
from math import isfinite

from sandbox.market_state.discovery_dataset import DiscoveryDataset, column_values, field_role

ML_DISCOVERY_MATRIX_VERSION = "ML_DISCOVERY_MATRIX_V1"


def _numeric(value, *, stored=False):
    if value is None:
        return None
    if type(value) not in ((float,) if stored else (int, float)):
        raise ValueError("numeric cells must be finite numbers or None")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("numeric value cannot be represented as finite float") from exc
    if not isfinite(number):
        raise ValueError("numeric cells must be finite")
    return 0.0 if number == 0 else number


def _categorical(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return _numeric(value, stored=True)
    raise ValueError("unsupported categorical cell")


def _indices(values, *, nonempty):
    if not isinstance(values, tuple) or (nonempty and not values):
        raise ValueError("row indices must be tuples with the required cardinality")
    if any(type(i) is not int or i < 0 for i in values):
        raise ValueError("row indices must be nonnegative integers")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("row indices must be strictly increasing")


@dataclass(frozen=True)
class MLFeatureSpec:
    field_id: str
    feature_kind: str

    def __post_init__(self):
        if not isinstance(self.field_id, str) or not self.field_id.startswith("x."):
            raise ValueError("feature field must be in x. namespace")
        if type(self.feature_kind) is not str or self.feature_kind not in ("NUMERIC", "CATEGORICAL"):
            raise ValueError("unsupported feature kind")


def _specs(specs):
    if not isinstance(specs, tuple) or not specs or any(not isinstance(s, MLFeatureSpec) for s in specs):
        raise ValueError("feature specs must be a nonempty tuple of MLFeatureSpec")
    for spec in specs:
        spec.__post_init__()
    if len({s.field_id for s in specs}) != len(specs):
        raise ValueError("duplicate feature field IDs")


def _fingerprint(version, source_row_indices, feature_specs, x_rows, target_field_id,
                 y_values, target_observed, observed_target_row_indices):
    tagged_rows = tuple(tuple(
        ["missing" if value is None else type(value).__name__, value]
        if spec.feature_kind == "CATEGORICAL" else value
        for spec, value in zip(feature_specs, row)) for row in x_rows)
    payload = {
        "version": version,
        "source_row_indices": source_row_indices,
        "feature_specs": [{"field_id": s.field_id, "feature_kind": s.feature_kind} for s in feature_specs],
        "x_rows": tagged_rows,
        "target_field_id": target_field_id,
        "y_values": y_values,
        "target_observed": target_observed,
        "observed_target_row_indices": observed_target_row_indices,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MLDiscoveryMatrix:
    version: str
    source_row_indices: tuple[int, ...]
    feature_specs: tuple[MLFeatureSpec, ...]
    x_rows: tuple[tuple[object, ...], ...]
    target_field_id: str
    y_values: tuple[float | None, ...]
    target_observed: tuple[bool, ...]
    observed_target_row_indices: tuple[int, ...]
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != ML_DISCOVERY_MATRIX_VERSION:
            raise ValueError("unsupported matrix version")
        _indices(self.source_row_indices, nonempty=True)
        _specs(self.feature_specs)
        size = len(self.source_row_indices)
        if not isinstance(self.x_rows, tuple) or len(self.x_rows) != size:
            raise ValueError("x_rows must match source row count")
        normalized_rows = []
        for row in self.x_rows:
            if not isinstance(row, tuple) or len(row) != len(self.feature_specs):
                raise ValueError("x row must be a tuple matching feature width")
            normalized_rows.append(tuple(_numeric(v, stored=True) if s.feature_kind == "NUMERIC" else _categorical(v)
                                         for s, v in zip(self.feature_specs, row)))
        if not isinstance(self.target_field_id, str) or not self.target_field_id.startswith("y."):
            raise ValueError("target field must be in y. namespace")
        if not isinstance(self.y_values, tuple) or len(self.y_values) != size:
            raise ValueError("y_values must match source row count")
        normalized_y = tuple(_numeric(v, stored=True) for v in self.y_values)
        if not isinstance(self.target_observed, tuple) or len(self.target_observed) != size or any(
                type(v) is not bool for v in self.target_observed):
            raise ValueError("target mask must be a tuple of exact bool values")
        if self.target_observed != tuple(v is not None for v in normalized_y):
            raise ValueError("target mask does not match missingness")
        _indices(self.observed_target_row_indices, nonempty=False)
        expected_indices = tuple(i for i, observed in zip(self.source_row_indices, self.target_observed) if observed)
        if self.observed_target_row_indices != expected_indices:
            raise ValueError("observed target indices must refer to original dataset rows")
        object.__setattr__(self, "x_rows", tuple(normalized_rows))
        object.__setattr__(self, "y_values", normalized_y)
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64 or any(
                c not in "0123456789abcdef" for c in self.fingerprint):
            raise ValueError("fingerprint must be lowercase hexadecimal SHA-256")
        expected = _fingerprint(self.version, self.source_row_indices, self.feature_specs, self.x_rows,
                                self.target_field_id, self.y_values, self.target_observed, self.observed_target_row_indices)
        if self.fingerprint != expected:
            raise ValueError("fingerprint does not match matrix content")


def build_ml_discovery_matrix(dataset: DiscoveryDataset, feature_specs, target_field_id: str,
                              row_indices=None) -> MLDiscoveryMatrix:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not dataset.rows:
        raise ValueError("ML matrix requires a nonempty dataset")
    specs = tuple(feature_specs)
    _specs(specs)
    for spec in specs:
        if field_role(dataset, spec.field_id) != "predictor":
            raise ValueError("ML features must have predictor role")
    if not isinstance(target_field_id, str):
        raise TypeError("target field must be string")
    if field_role(dataset, target_field_id) != "target":
        raise ValueError("prediction target must have target role")
    indices = tuple(range(len(dataset.rows))) if row_indices is None else tuple(row_indices)
    _indices(indices, nonempty=True)
    if indices[-1] >= len(dataset.rows):
        raise ValueError("source row index exceeds dataset bounds")
    columns = tuple(column_values(dataset, s.field_id) for s in specs)
    target = column_values(dataset, target_field_id)
    rows = tuple(tuple(_numeric(column[i]) if spec.feature_kind == "NUMERIC" else _categorical(column[i])
                       for spec, column in zip(specs, columns)) for i in indices)
    values = tuple(_numeric(target[i]) for i in indices)
    observed = tuple(v is not None for v in values)
    observed_indices = tuple(i for i, present in zip(indices, observed) if present)
    content = (ML_DISCOVERY_MATRIX_VERSION, indices, specs, rows, target_field_id, values, observed, observed_indices)
    return MLDiscoveryMatrix(*content, _fingerprint(*content))
