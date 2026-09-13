"""Deterministic descriptive comparison of two disjoint target cohorts."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isclose, isfinite
from statistics import fmean, median

from sandbox.market_state.discovery_dataset import DiscoveryDataset, column_values, field_role


def _indices(values):
    if not isinstance(values, tuple) or not values or any(type(i) is not int or i < 0 for i in values):
        raise ValueError("indices must be a nonempty tuple of nonnegative integers")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("indices must be strictly increasing")


def _finite(value):
    return type(value) is float and isfinite(value)


@dataclass(frozen=True)
class NumericCohortComparison:
    field_id: str
    left_row_indices: tuple[int, ...]
    right_row_indices: tuple[int, ...]
    left_row_count: int
    right_row_count: int
    left_observed_count: int
    right_observed_count: int
    left_missing_count: int
    right_missing_count: int
    left_mean: float | None
    right_mean: float | None
    mean_difference: float | None
    left_median: float | None
    right_median: float | None
    median_difference: float | None
    pair_count: int
    left_greater_count: int
    tie_count: int
    left_less_count: int
    probability_of_superiority: float | None
    cliffs_delta: float | None

    def __post_init__(self):
        if not isinstance(self.field_id, str) or not self.field_id.strip():
            raise ValueError("field_id must be nonempty string")
        _indices(self.left_row_indices)
        _indices(self.right_row_indices)
        if set(self.left_row_indices).intersection(self.right_row_indices):
            raise ValueError("cohorts must be disjoint")
        for side in ("left", "right"):
            rows, observed, missing = (getattr(self, side + suffix) for suffix in
                                       ("_row_count", "_observed_count", "_missing_count"))
            if any(type(v) is not int or v < 0 for v in (rows, observed, missing)):
                raise ValueError("counts must be nonnegative integers")
            if rows <= 0 or rows != len(getattr(self, side + "_row_indices")) or observed + missing != rows:
                raise ValueError("inconsistent row counts")
            for suffix in ("_mean", "_median"):
                value = getattr(self, side + suffix)
                if (observed == 0 and value is not None) or (observed > 0 and not _finite(value)):
                    raise ValueError("location statistics must match observed count")
        for name in ("mean", "median"):
            left, right, difference = (
                getattr(self, "left_" + name), getattr(self, "right_" + name), getattr(self, name + "_difference"))
            if left is None or right is None:
                if difference is not None:
                    raise ValueError("difference requires observations on both sides")
            elif not _finite(difference) or difference != left - right:
                raise ValueError("inconsistent location difference")
        counts = (self.pair_count, self.left_greater_count, self.tie_count, self.left_less_count)
        if any(type(v) is not int or v < 0 for v in counts):
            raise ValueError("pair counts must be nonnegative integers")
        if sum(counts[1:]) != self.pair_count or self.pair_count != self.left_observed_count * self.right_observed_count:
            raise ValueError("inconsistent pair counts")
        p, delta = self.probability_of_superiority, self.cliffs_delta
        if not self.pair_count:
            if p is not None or delta is not None:
                raise ValueError("rank statistics require observed pairs")
        else:
            expected_p = (self.left_greater_count + .5 * self.tie_count) / self.pair_count
            expected_delta = (self.left_greater_count - self.left_less_count) / self.pair_count
            if not _finite(p) or not 0 <= p <= 1 or not _finite(delta) or not -1 <= delta <= 1:
                raise ValueError("invalid rank statistics")
            if p != expected_p or delta != expected_delta or not isclose(delta, 2 * p - 1, abs_tol=1e-15):
                raise ValueError("inconsistent rank statistics")


def compare_cohort_rows(dataset: DiscoveryDataset, left_row_indices, right_row_indices,
                        field_id: str) -> NumericCohortComparison:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not isinstance(field_id, str):
        raise TypeError("field_id must be string")
    if field_role(dataset, field_id) != "target":
        raise ValueError("only target fields may be compared")
    left_indices, right_indices = tuple(left_row_indices), tuple(right_row_indices)
    for indices in (left_indices, right_indices):
        _indices(indices)
        if indices[-1] >= len(dataset.rows):
            raise ValueError("row index out of bounds")
    if set(left_indices).intersection(right_indices):
        raise ValueError("cohorts must be disjoint")
    column = column_values(dataset, field_id)
    observed = []
    for indices in (left_indices, right_indices):
        values = []
        for i in indices:
            value = column[i]
            if value is None:
                continue
            if type(value) not in (int, float):
                raise ValueError("observed targets must be finite numeric values")
            try:
                number = float(value)
            except OverflowError as exc:
                raise ValueError("observed target exceeds finite float range") from exc
            if not isfinite(number):
                raise ValueError("observed targets must be finite")
            values.append(number)
        observed.append(sorted(values))
    left, right = observed
    lm, rm = (fmean(v) if v else None for v in observed)
    ld, rd = (float(median(v)) if v else None for v in observed)
    pairs = len(left) * len(right)
    greater = ties = 0
    for value in left:
        low, high = bisect_left(right, value), bisect_right(right, value)
        greater += low
        ties += high - low
    less = pairs - greater - ties
    return NumericCohortComparison(field_id, left_indices, right_indices,
        len(left_indices), len(right_indices), len(left), len(right),
        len(left_indices) - len(left), len(right_indices) - len(right),
        lm, rm, lm - rm if lm is not None and rm is not None else None,
        ld, rd, ld - rd if ld is not None and rd is not None else None,
        pairs, greater, ties, less, (greater + .5 * ties) / pairs if pairs else None,
        (greater - less) / pairs if pairs else None)
