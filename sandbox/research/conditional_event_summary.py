"""Caller-selected categorical conditions over authoritative event cohorts."""
from dataclasses import dataclass
from sandbox.market_state.discovery_dataset import DiscoveryDataset, assemble_discovery_dataset, column_values, field_role
from .event_cohort_summary import summarize_event_cohorts
from .outcome_distribution import NumericOutcomeDistribution, summarize_outcomes

_EVENT_FIELDS = ("x.anchor_kind", "x.event.pattern_code", "x.event.checkpoint_bars")


def _typed(value):
    if value is None:
        return (0, 0)
    if type(value) is bool:
        return (1, int(value))
    if type(value) is int:
        return (2, value)
    if type(value) is str:
        return (3, value)
    raise ValueError("condition values must be None, bool, int, or str")


def _fields(fields):
    if not fields or any(not isinstance(f, str) for f in fields):
        raise ValueError("at least one string condition field is required")
    if len(set(fields)) != len(fields) or any(f in _EVENT_FIELDS for f in fields):
        raise ValueError("duplicate or event-definition condition field")


@dataclass(frozen=True)
class ConditionalEventCohortSummary:
    event_cohort_id: str
    anchor_kind: str
    pattern_code: str | None
    checkpoint_bars: int | None
    condition_field_ids: tuple[str, ...]
    condition_values: tuple[None | bool | int | str, ...]
    row_indices: tuple[int, ...]
    row_count: int
    outcome_distributions: tuple[NumericOutcomeDistribution, ...]

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.event_cohort_id, self.anchor_kind)):
            raise ValueError("event identity must be nonempty strings")
        if not isinstance(self.condition_field_ids, tuple):
            raise ValueError("condition_field_ids must be tuple")
        _fields(self.condition_field_ids)
        if not isinstance(self.condition_values, tuple) or len(self.condition_values) != len(self.condition_field_ids):
            raise ValueError("condition fields and values must have equal lengths")
        for value in self.condition_values:
            _typed(value)
        if not isinstance(self.row_indices, tuple) or any(type(i) is not int or i < 0 for i in self.row_indices):
            raise ValueError("row_indices must be tuple of nonnegative integers")
        if any(a >= b for a, b in zip(self.row_indices, self.row_indices[1:])):
            raise ValueError("row_indices must be strictly increasing")
        if type(self.row_count) is not int or self.row_count <= 0 or self.row_count != len(self.row_indices):
            raise ValueError("inconsistent row_count")
        if not isinstance(self.outcome_distributions, tuple) or any(
                not isinstance(d, NumericOutcomeDistribution) or d.row_count != self.row_count for d in self.outcome_distributions):
            raise ValueError("invalid outcome distributions")
        if len({d.field_id for d in self.outcome_distributions}) != len(self.outcome_distributions):
            raise ValueError("duplicate distribution fields")


def summarize_conditioned_event_cohorts(dataset: DiscoveryDataset, condition_field_ids,
                                        field_ids=None) -> tuple[ConditionalEventCohortSummary, ...]:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not dataset.rows:
        return ()
    conditions = tuple(condition_field_ids)
    _fields(conditions)
    for field in conditions:
        if field_role(dataset, field) != "predictor":
            raise ValueError("condition must be predictor")
    fields = dataset.target_field_ids if field_ids is None else tuple(field_ids)
    if any(not isinstance(f, str) for f in fields):
        raise TypeError("target field IDs must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate target fields")
    columns = tuple(column_values(dataset, field) for field in conditions)
    for column in columns:
        for value in column:
            _typed(value)
    events = summarize_event_cohorts(dataset, field_ids=())
    results = []
    for event in sorted(events, key=lambda e: e.cohort_id):
        groups = {}
        for index in event.row_indices:
            values = tuple(column[index] for column in columns)
            key = tuple(_typed(value) for value in values)
            if key not in groups:
                groups[key] = (values, [])
            groups[key][1].append(index)
        for key in sorted(groups):
            values, indices = groups[key]
            subset = assemble_discovery_dataset(tuple(dataset.rows[i] for i in indices))
            distributions = summarize_outcomes(subset, fields)
            results.append(ConditionalEventCohortSummary(event.cohort_id, event.anchor_kind,
                event.pattern_code, event.checkpoint_bars, conditions, values, tuple(indices), len(indices), distributions))
    return tuple(results)


def summarize_single_conditioned_event_cohorts(dataset: DiscoveryDataset, condition_field_id: str, field_ids=None):
    return summarize_conditioned_event_cohorts(dataset, (condition_field_id,), field_ids)
