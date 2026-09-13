"""Role-neutral arithmetic evaluation of matching prediction artifacts."""
from dataclasses import dataclass, fields
import hashlib
import json
from math import copysign, fsum, isfinite, sqrt
from statistics import fmean, median

from .ml_preprocessing import EncodedMLMatrix
from .ml_lightgbm import LightGBMPredictionBatch

EVALUATION_VERSION = "ML_PREDICTION_EVALUATION_V1"
_REQUIRED_METRICS = (
    "actual_mean", "actual_median", "prediction_mean", "prediction_median",
    "residual_mean", "residual_median", "mean_absolute_error", "median_absolute_error",
    "root_mean_squared_error", "mean_squared_error", "actual_minimum", "actual_maximum",
    "prediction_minimum", "prediction_maximum",
)
_OPTIONAL_METRICS = ("r_squared", "pearson_correlation", "spearman_rank_correlation")


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")


def _digest(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _clean(value):
    if value is None:
        return None
    value = float(value)
    if not isfinite(value):
        raise ValueError("evaluation metrics must be finite")
    return 0.0 if value == 0 else value


@dataclass(frozen=True)
class MLPredictionEvaluation:
    version: str
    model_fingerprint: str
    encoded_matrix_fingerprint: str
    prediction_batch_fingerprint: str
    target_field_id: str
    source_row_count: int
    observed_target_count: int
    missing_target_count: int
    observed_source_row_indices: tuple[int, ...]
    actual_mean: float
    actual_median: float
    prediction_mean: float
    prediction_median: float
    residual_mean: float
    residual_median: float
    mean_absolute_error: float
    median_absolute_error: float
    root_mean_squared_error: float
    mean_squared_error: float
    r_squared: float | None
    pearson_correlation: float | None
    spearman_rank_correlation: float | None
    actual_minimum: float
    actual_maximum: float
    prediction_minimum: float
    prediction_maximum: float
    fingerprint: str

    def __post_init__(self):
        if type(self.version) is not str or self.version != EVALUATION_VERSION:
            raise ValueError("unsupported evaluation version")
        for value in (self.model_fingerprint, self.encoded_matrix_fingerprint, self.prediction_batch_fingerprint, self.fingerprint):
            _sha(value)
        if not isinstance(self.target_field_id, str) or not self.target_field_id.startswith("y."):
            raise ValueError("target must be in y. namespace")
        for value, minimum in ((self.source_row_count, 2), (self.observed_target_count, 2), (self.missing_target_count, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError("invalid evaluation count")
        if self.observed_target_count + self.missing_target_count != self.source_row_count:
            raise ValueError("inconsistent evaluation counts")
        indices = self.observed_source_row_indices
        if not isinstance(indices, tuple) or len(indices) != self.observed_target_count or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("invalid observed source indices")
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError("observed source indices must be strictly increasing")
        for name in _REQUIRED_METRICS + _OPTIONAL_METRICS:
            value = getattr(self, name)
            if value is None and name in _OPTIONAL_METRICS:
                continue
            if type(value) is not float or not isfinite(value) or (value == 0 and copysign(1.0, value) < 0):
                raise ValueError("metrics must be finite exact floats without negative zero")
        for name in ("mean_absolute_error", "median_absolute_error", "mean_squared_error", "root_mean_squared_error"):
            if getattr(self, name) < 0:
                raise ValueError("error magnitudes must be nonnegative")
        for value in (self.pearson_correlation, self.spearman_rank_correlation):
            if value is not None and not -1 <= value <= 1:
                raise ValueError("correlations must be in [-1, 1]")
        if self.actual_minimum > self.actual_maximum or self.prediction_minimum > self.prediction_maximum:
            raise ValueError("minimum exceeds maximum")
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "fingerprint"}
        if self.fingerprint != _digest(payload):
            raise ValueError("evaluation fingerprint mismatch")


def _average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return tuple(ranks)


def _pearson(left, right):
    mean_left, mean_right = fmean(left), fmean(right)
    a = tuple(v - mean_left for v in left)
    b = tuple(v - mean_right for v in right)
    scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
    if scale_a == 0 or scale_b == 0:
        return None
    # Positive rescaling cancels in the ratio and avoids variance-product overflow.
    a = tuple(v / scale_a for v in a)
    b = tuple(v / scale_b for v in b)
    value = fsum(x * y for x, y in zip(a, b)) / sqrt(fsum(x * x for x in a) * fsum(y * y for y in b))
    if abs(value) > 1:
        if abs(value) - 1 > 1e-15:
            raise ValueError("correlation exceeds arithmetic tolerance")
        value = copysign(1.0, value)
    return _clean(value)


def evaluate_prediction_batch(matrix: EncodedMLMatrix, predictions: LightGBMPredictionBatch) -> MLPredictionEvaluation:
    if not isinstance(matrix, EncodedMLMatrix) or not isinstance(predictions, LightGBMPredictionBatch):
        raise TypeError("expected EncodedMLMatrix and LightGBMPredictionBatch")
    if (predictions.encoded_matrix_fingerprint != matrix.fingerprint or
        predictions.source_row_indices != matrix.source_row_indices or predictions.target_field_id != matrix.target_field_id):
        raise ValueError("prediction artifact must match the exact encoded matrix")
    positions = tuple(i for i, observed in enumerate(matrix.target_observed) if observed)
    if len(positions) < 2:
        raise ValueError("evaluation requires at least two observed targets")
    actual = tuple(matrix.y_values[i] for i in positions)
    predicted = tuple(predictions.predictions[i] for i in positions)
    try:
        residuals = tuple(p - a for a, p in zip(actual, predicted))
        absolute = tuple(abs(r) for r in residuals)
        squared = tuple(r ** 2 for r in residuals)
        actual_mean = fmean(actual)
        mse = fmean(squared)
        total = fsum((a - actual_mean) ** 2 for a in actual)
        metrics = dict(actual_mean=actual_mean, actual_median=median(actual),
            prediction_mean=fmean(predicted), prediction_median=median(predicted),
            residual_mean=fmean(residuals), residual_median=median(residuals),
            mean_absolute_error=fmean(absolute), median_absolute_error=median(absolute),
            root_mean_squared_error=sqrt(mse), mean_squared_error=mse,
            r_squared=None if total == 0 else 1 - fsum(squared) / total,
            pearson_correlation=_pearson(actual, predicted),
            spearman_rank_correlation=_pearson(_average_ranks(actual), _average_ranks(predicted)),
            actual_minimum=min(actual), actual_maximum=max(actual),
            prediction_minimum=min(predicted), prediction_maximum=max(predicted))
    except OverflowError as exc:
        raise ValueError("evaluation arithmetic exceeds finite float range") from exc
    state = dict(version=EVALUATION_VERSION, model_fingerprint=predictions.model_fingerprint,
        encoded_matrix_fingerprint=matrix.fingerprint, prediction_batch_fingerprint=predictions.fingerprint,
        target_field_id=matrix.target_field_id, source_row_count=len(matrix.source_row_indices),
        observed_target_count=len(positions), missing_target_count=len(matrix.source_row_indices) - len(positions),
        observed_source_row_indices=matrix.observed_target_row_indices,
        **{name: _clean(value) for name, value in metrics.items()})
    return MLPredictionEvaluation(**state, fingerprint=_digest(state))
