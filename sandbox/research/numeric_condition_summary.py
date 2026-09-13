"""Explicit numeric bins over authoritative event cohorts."""
from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite

from sandbox.market_state.discovery_dataset import DiscoveryDataset, assemble_discovery_dataset, column_values, field_role
from .event_cohort_summary import summarize_event_cohorts
from .outcome_distribution import NumericOutcomeDistribution, summarize_outcomes

_EVENT_FIELDS = ("x.anchor_kind", "x.event.pattern_code", "x.event.checkpoint_bars")


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("numeric values must be finite int or float")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("numeric value exceeds finite float range") from exc
    if not isfinite(result):
        raise ValueError("numeric values must be finite")
    return result


@dataclass(frozen=True)
class NumericConditionalEventCohortSummary:
    event_cohort_id: str
    anchor_kind: str
    pattern_code: str | None
    checkpoint_bars: int | None
    condition_field_id: str
    missing: bool
    bin_index: int | None
    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None
    row_indices: tuple[int, ...]
    row_count: int
    outcome_distributions: tuple[NumericOutcomeDistribution, ...]

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.event_cohort_id, self.anchor_kind, self.condition_field_id)):
            raise ValueError("identity fields must be nonempty strings")
        if type(self.missing) is not bool:
            raise ValueError("missing must be bool")
        if not isinstance(self.row_indices, tuple) or any(type(i) is not int or i < 0 for i in self.row_indices):
            raise ValueError("row indices must be a tuple of nonnegative integers")
        if any(a >= b for a, b in zip(self.row_indices, self.row_indices[1:])):
            raise ValueError("row indices must be strictly increasing")
        if type(self.row_count) is not int or self.row_count <= 0 or self.row_count != len(self.row_indices):
            raise ValueError("inconsistent row count")
        if not isinstance(self.outcome_distributions, tuple) or any(
                not isinstance(d, NumericOutcomeDistribution) or d.row_count != self.row_count
                for d in self.outcome_distributions):
            raise ValueError("invalid outcome distributions")
        if len({d.field_id for d in self.outcome_distributions}) != len(self.outcome_distributions):
            raise ValueError("duplicate distribution fields")
        metadata = (self.bin_index, self.lower_bound, self.upper_bound, self.lower_inclusive, self.upper_inclusive)
        if self.missing:
            if any(v is not None for v in metadata):
                raise ValueError("missing group cannot have interval metadata")
            return
        if type(self.bin_index) is not int or self.bin_index < 0:
            raise ValueError("bin index must be nonnegative integer")
        lo, hi = self.lower_bound, self.upper_bound
        if any(v is not None and (type(v) is not float or not isfinite(v)) for v in (lo, hi)):
            raise ValueError("bounds must be finite floats or None")
        if lo is None:
            valid = self.bin_index == 0 and hi is not None and self.lower_inclusive is None and self.upper_inclusive is False
        elif hi is None:
            valid = self.bin_index > 0 and self.lower_inclusive is True and self.upper_inclusive is None
        else:
            valid = self.bin_index > 0 and lo < hi and self.lower_inclusive is True and self.upper_inclusive is False
        if not valid:
            raise ValueError("unsupported interval metadata")


def summarize_numeric_conditioned_event_cohorts(dataset: DiscoveryDataset, condition_field_id: str,
                                                cutpoints, field_ids=None) -> tuple[NumericConditionalEventCohortSummary, ...]:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not dataset.rows:
        return ()
    if not isinstance(condition_field_id, str):
        raise TypeError("condition field must be string")
    if condition_field_id in _EVENT_FIELDS or field_role(dataset, condition_field_id) != "predictor":
        raise ValueError("condition must be a non-event predictor")
    cuts = tuple(cutpoints)
    cuts = tuple(_number(c) for c in cuts)
    if not cuts or any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError("cutpoints must be nonempty and strictly increasing")
    fields = dataset.target_field_ids if field_ids is None else tuple(field_ids)
    if any(not isinstance(f, str) for f in fields):
        raise TypeError("target field IDs must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate target field IDs")
    values = column_values(dataset, condition_field_id)
    bins = tuple(-1 if v is None else bisect_right(cuts, _number(v)) for v in values)
    events = summarize_event_cohorts(dataset, field_ids=())
    results = []
    for event in sorted(events, key=lambda e: e.cohort_id):
        groups = {}
        for index in event.row_indices:
            groups.setdefault(bins[index], []).append(index)
        for key in sorted(groups):
            indices = tuple(groups[key])
            subset = assemble_discovery_dataset(tuple(dataset.rows[i] for i in indices))
            distributions = summarize_outcomes(subset, fields)
            missing = key == -1
            lo = cuts[key - 1] if key > 0 else None
            hi = cuts[key] if 0 <= key < len(cuts) else None
            results.append(NumericConditionalEventCohortSummary(
                event.cohort_id, event.anchor_kind, event.pattern_code, event.checkpoint_bars,
                condition_field_id, missing, None if missing else key, lo, hi,
                True if lo is not None else None, False if hi is not None else None,
                indices, len(indices), distributions))
    return tuple(results)
