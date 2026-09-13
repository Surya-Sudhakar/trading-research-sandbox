"""Exact causal event cohorts with delegated target distributions."""
from dataclasses import dataclass
from sandbox.market_state.discovery_dataset import DiscoveryDataset, assemble_discovery_dataset, column_values, field_role
from .outcome_distribution import NumericOutcomeDistribution, summarize_outcomes


def _cohort_id(kind, pattern, checkpoint):
    if kind == "TOUCH_TRANSITION":
        if not isinstance(pattern, str) or not pattern.strip() or checkpoint is not None:
            raise ValueError("invalid touch event combination")
        return f"TOUCH_TRANSITION__{pattern}"
    if kind == "UNTOUCHED_CHECKPOINT":
        if pattern is not None or type(checkpoint) is not int or checkpoint <= 0:
            raise ValueError("invalid untouched event combination")
        return f"UNTOUCHED_CHECKPOINT__{checkpoint}"
    raise ValueError("unknown anchor kind")


@dataclass(frozen=True)
class EventCohortSummary:
    cohort_id: str
    anchor_kind: str
    pattern_code: str | None
    checkpoint_bars: int | None
    row_indices: tuple[int, ...]
    row_count: int
    outcome_distributions: tuple[NumericOutcomeDistribution, ...]

    def __post_init__(self):
        if self.cohort_id != _cohort_id(self.anchor_kind, self.pattern_code, self.checkpoint_bars):
            raise ValueError("inconsistent cohort_id")
        if not isinstance(self.row_indices, tuple) or any(type(i) is not int or i < 0 for i in self.row_indices):
            raise ValueError("row_indices must be a tuple of nonnegative integers")
        if any(a >= b for a, b in zip(self.row_indices, self.row_indices[1:])):
            raise ValueError("row_indices must be strictly increasing")
        if type(self.row_count) is not int or self.row_count <= 0 or self.row_count != len(self.row_indices):
            raise ValueError("inconsistent row_count")
        distributions = tuple(self.outcome_distributions)
        if any(not isinstance(d, NumericOutcomeDistribution) or d.row_count != self.row_count for d in distributions):
            raise ValueError("invalid outcome distribution or row_count")
        if len({d.field_id for d in distributions}) != len(distributions):
            raise ValueError("duplicate outcome field IDs")
        object.__setattr__(self, "outcome_distributions", distributions)


def summarize_event_cohorts(dataset: DiscoveryDataset, field_ids=None) -> tuple[EventCohortSummary, ...]:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not dataset.rows:
        return ()
    fields = dataset.target_field_ids if field_ids is None else tuple(field_ids)
    if any(not isinstance(field, str) for field in fields):
        raise TypeError("field IDs must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate field IDs")
    required = ("x.anchor_kind", "x.event.pattern_code", "x.event.checkpoint_bars")
    for field in required:
        if field_role(dataset, field) != "predictor":
            raise ValueError("event fields must be predictors")
    columns = tuple(column_values(dataset, field) for field in required)
    groups = {}
    for index, (kind, pattern, checkpoint) in enumerate(zip(*columns)):
        cohort_id = _cohort_id(kind, pattern, checkpoint)
        if cohort_id not in groups:
            groups[cohort_id] = (kind, pattern, checkpoint, [])
        groups[cohort_id][3].append(index)
    results = []
    for cohort_id in sorted(groups):
        kind, pattern, checkpoint, indices = groups[cohort_id]
        subset = assemble_discovery_dataset(tuple(dataset.rows[i] for i in indices))
        distributions = summarize_outcomes(subset, fields)
        results.append(EventCohortSummary(cohort_id, kind, pattern, checkpoint,
                                          tuple(indices), len(indices), distributions))
    return tuple(results)
