"""Deterministic descriptive distributions of numeric target columns only."""
from dataclasses import dataclass
from math import floor, ceil, isfinite
from statistics import fmean

from sandbox.market_state.discovery_dataset import DiscoveryDataset, column_values, field_role


@dataclass(frozen=True)
class NumericOutcomeDistribution:
    field_id: str
    row_count: int
    observed_count: int
    missing_count: int
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None
    mean: float | None
    negative_count: int
    zero_count: int
    positive_count: int
    negative_fraction: float | None
    zero_fraction: float | None
    positive_fraction: float | None

    def __post_init__(self):
        counts = (self.row_count, self.observed_count, self.missing_count,
                  self.negative_count, self.zero_count, self.positive_count)
        if any(type(c) is not int or c < 0 for c in counts):
            raise ValueError("counts must be nonnegative integers")
        signs = (self.negative_count, self.zero_count, self.positive_count)
        if self.observed_count + self.missing_count != self.row_count or sum(signs) != self.observed_count:
            raise ValueError("inconsistent counts")
        stats = (self.minimum, self.q25, self.median, self.q75, self.maximum, self.mean)
        fractions = (self.negative_fraction, self.zero_fraction, self.positive_fraction)
        if self.observed_count == 0:
            if any(v is not None for v in stats + fractions):
                raise ValueError("empty observations require None statistics and fractions")
        else:
            if any(type(v) is not float or not isfinite(v) for v in stats):
                raise ValueError("statistics must be finite floats")
            if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 <= f <= 1
                   or f != c / self.observed_count for f, c in zip(fractions, signs)):
                raise ValueError("inconsistent sign fractions")


def _quantile(values, q):
    position = (len(values) - 1) * q
    lower, upper = floor(position), ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def summarize_outcome(dataset: DiscoveryDataset, field_id: str) -> NumericOutcomeDistribution:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not isinstance(field_id, str):
        raise TypeError("field_id must be string")
    if field_role(dataset, field_id) != "target":
        raise ValueError("only target fields may be summarized as outcomes")
    values = column_values(dataset, field_id)
    observed = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("observed outcomes must be finite numeric values")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("observed outcomes must be finite") from exc
        if not isfinite(number):
            raise ValueError("observed outcomes must be finite")
        observed.append(number)
    observed.sort()
    count = len(observed)
    signs = (sum(v < 0 for v in observed), sum(v == 0 for v in observed), sum(v > 0 for v in observed))
    stats = (observed[0], _quantile(observed, .25), _quantile(observed, .5),
             _quantile(observed, .75), observed[-1], fmean(observed)) if count else (None,) * 6
    fractions = tuple(c / count for c in signs) if count else (None,) * 3
    return NumericOutcomeDistribution(field_id, len(values), count, len(values) - count,
                                      *stats, *signs, *fractions)


def summarize_outcomes(dataset: DiscoveryDataset, field_ids=None) -> tuple[NumericOutcomeDistribution, ...]:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    fields = dataset.target_field_ids if field_ids is None else tuple(field_ids)
    if any(not isinstance(field, str) for field in fields):
        raise TypeError("field IDs must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate field IDs")
    return tuple(summarize_outcome(dataset, field) for field in fields)
