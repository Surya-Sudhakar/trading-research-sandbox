"""Two observed contexts intersected within authoritative event partitions."""
from dataclasses import dataclass
from math import isfinite

from sandbox.market_state.discovery_dataset import DiscoveryDataset, field_role
from .conditional_event_summary import summarize_single_conditioned_event_cohorts
from .numeric_condition_summary import NumericConditionalEventCohortSummary, summarize_numeric_conditioned_event_cohorts
from .cohort_comparison import NumericCohortComparison, compare_cohort_rows


def _indices(indices):
    if not isinstance(indices, tuple) or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("indices must be tuples of nonnegative integers")
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("indices must be strictly increasing")


@dataclass(frozen=True)
class ContextCondition:
    context_kind: str
    condition_field_id: str
    categorical_value: None | bool | int | str
    numeric_cutpoints: tuple[float, ...] | None
    numeric_missing: bool | None
    numeric_bin_index: int | None
    numeric_lower_bound: float | None
    numeric_upper_bound: float | None
    numeric_lower_inclusive: bool | None
    numeric_upper_inclusive: bool | None

    def __post_init__(self):
        if type(self.context_kind) is not str or self.context_kind not in ("CATEGORICAL", "NUMERIC"):
            raise ValueError("unsupported context kind")
        if not isinstance(self.condition_field_id, str) or not self.condition_field_id.strip():
            raise ValueError("condition field must be nonempty string")
        metadata = (self.numeric_missing, self.numeric_bin_index, self.numeric_lower_bound,
                    self.numeric_upper_bound, self.numeric_lower_inclusive, self.numeric_upper_inclusive)
        if self.context_kind == "CATEGORICAL":
            if self.numeric_cutpoints is not None or any(v is not None for v in metadata):
                raise ValueError("categorical context cannot contain numeric metadata")
            if self.categorical_value is not None and type(self.categorical_value) not in (bool, int, str):
                raise ValueError("unsupported categorical value")
            return
        cuts = self.numeric_cutpoints
        if not isinstance(cuts, tuple) or not cuts or any(type(c) is not float or not isfinite(c) for c in cuts):
            raise ValueError("cutpoints must be a nonempty tuple of finite floats")
        if any(a >= b for a, b in zip(cuts, cuts[1:])) or self.categorical_value is not None:
            raise ValueError("invalid numeric context identity")
        # Reuse the established interval validator with a minimal validation-only row.
        NumericConditionalEventCohortSummary("CONTEXT", "CONTEXT", None, None,
            self.condition_field_id, *metadata, (0,), 1, ())
        if not self.numeric_missing:
            index = self.numeric_bin_index
            if index > len(cuts):
                raise ValueError("bin exceeds cutpoints")
            lower = cuts[index - 1] if index else None
            upper = cuts[index] if index < len(cuts) else None
            if self.numeric_lower_bound != lower or self.numeric_upper_bound != upper:
                raise ValueError("interval does not match cutpoints")


@dataclass(frozen=True)
class TwoContextScanResult:
    left_condition: ContextCondition
    right_condition: ContextCondition
    event_cohort_id: str
    anchor_kind: str
    pattern_code: str | None
    checkpoint_bars: int | None
    left_row_indices: tuple[int, ...]
    right_row_indices: tuple[int, ...]
    joint_row_indices: tuple[int, ...]
    event_complement_row_indices: tuple[int, ...]
    left_without_right_row_indices: tuple[int, ...]
    right_without_left_row_indices: tuple[int, ...]
    target_field_id: str
    joint_vs_event_complement: NumericCohortComparison
    joint_vs_left_without_right: NumericCohortComparison | None
    joint_vs_right_without_left: NumericCohortComparison | None

    def __post_init__(self):
        if not isinstance(self.left_condition, ContextCondition) or not isinstance(self.right_condition, ContextCondition):
            raise ValueError("conditions must be ContextCondition")
        if self.left_condition.condition_field_id == self.right_condition.condition_field_id:
            raise ValueError("interaction requires distinct condition fields")
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.event_cohort_id, self.anchor_kind, self.target_field_id)):
            raise ValueError("event and target identity must be nonempty strings")
        if self.pattern_code is not None and (not isinstance(self.pattern_code, str) or not self.pattern_code.strip()):
            raise ValueError("invalid pattern code")
        if self.checkpoint_bars is not None and (type(self.checkpoint_bars) is not int or self.checkpoint_bars <= 0):
            raise ValueError("invalid checkpoint")
        indices = (self.left_row_indices, self.right_row_indices, self.joint_row_indices,
                   self.event_complement_row_indices, self.left_without_right_row_indices,
                   self.right_without_left_row_indices)
        for values in indices:
            _indices(values)
        left, right, joint, rest, left_only, right_only = map(set, indices)
        if not joint or not rest or joint & rest:
            raise ValueError("joint and event complement must be nonempty and disjoint")
        if joint != left & right or left_only != left - joint or right_only != right - joint:
            raise ValueError("inconsistent intersection membership")
        if not (left_only | right_only) <= rest:
            raise ValueError("event complement must contain all non-joint context rows")
        for rows, comparison in (
            (self.event_complement_row_indices, self.joint_vs_event_complement),
            (self.left_without_right_row_indices, self.joint_vs_left_without_right),
            (self.right_without_left_row_indices, self.joint_vs_right_without_left)):
            if not rows:
                if comparison is not None:
                    raise ValueError("empty baseline requires None comparison")
            elif not isinstance(comparison, NumericCohortComparison) or (
                comparison.left_row_indices != self.joint_row_indices or
                comparison.right_row_indices != rows or comparison.field_id != self.target_field_id):
                raise ValueError("comparison does not match stored membership")


def _source(kind, field, cuts, cohorts):
    events = {}
    for group in cohorts:
        metadata = (group.anchor_kind, group.pattern_code, group.checkpoint_bars)
        if group.event_cohort_id not in events:
            events[group.event_cohort_id] = (metadata, set(), [])
        previous, universe, groups = events[group.event_cohort_id]
        if previous != metadata:
            raise ValueError("inconsistent event metadata within source")
        universe.update(group.row_indices)
        if kind == "CATEGORICAL":
            condition = ContextCondition(kind, field, group.condition_values[0], None, *(None,) * 6)
        else:
            condition = ContextCondition(kind, field, None, cuts, group.missing, group.bin_index,
                group.lower_bound, group.upper_bound, group.lower_inclusive, group.upper_inclusive)
        groups.append((condition, group.row_indices, set(group.row_indices)))
    return field, events


def scan_two_context_interactions(dataset: DiscoveryDataset, categorical_field_ids=(),
                                  numeric_condition_specs=(), field_ids=None) -> tuple[TwoContextScanResult, ...]:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not dataset.rows:
        return ()
    categorical = tuple(categorical_field_ids)
    numeric = tuple(numeric_condition_specs)
    fields = dataset.target_field_ids if field_ids is None else tuple(field_ids)
    if any(not isinstance(f, str) for f in categorical + fields):
        raise TypeError("field IDs must be strings")
    if len(set(categorical)) != len(categorical) or len(set(fields)) != len(fields):
        raise ValueError("duplicate categorical or target field IDs")
    for field in fields:
        if field_role(dataset, field) != "target":
            raise ValueError("comparison field must be target")
    sources = []
    for field in categorical:
        cohorts = summarize_single_conditioned_event_cohorts(dataset, field, field_ids=())
        sources.append(_source("CATEGORICAL", field, None, cohorts))
    for field, cutpoints in numeric:
        cuts = tuple(cutpoints)
        cohorts = summarize_numeric_conditioned_event_cohorts(dataset, field, cuts, field_ids=())
        sources.append(_source("NUMERIC", field, tuple(float(c) for c in cuts), cohorts))
    results = []
    for position, (left_field, left_events) in enumerate(sources):
        for right_field, right_events in sources[position + 1:]:
            if left_field == right_field:
                continue
            for event in sorted(left_events.keys() & right_events.keys()):
                metadata, universe, left_groups = left_events[event]
                right_metadata, right_universe, right_groups = right_events[event]
                if metadata != right_metadata or universe != right_universe:
                    raise ValueError("inconsistent authoritative event partitions")
                for left_condition, left_rows, left_set in left_groups:
                    for right_condition, right_rows, right_set in right_groups:
                        joint_set = left_set & right_set
                        rest_set = universe - joint_set
                        if not joint_set or not rest_set:
                            continue
                        joint = tuple(sorted(joint_set))
                        rest = tuple(sorted(rest_set))
                        left_only = tuple(sorted(left_set - joint_set))
                        right_only = tuple(sorted(right_set - joint_set))
                        for target in fields:
                            event_comparison = compare_cohort_rows(dataset, joint, rest, target)
                            left_comparison = compare_cohort_rows(dataset, joint, left_only, target) if left_only else None
                            right_comparison = compare_cohort_rows(dataset, joint, right_only, target) if right_only else None
                            results.append(TwoContextScanResult(left_condition, right_condition, event, *metadata,
                                left_rows, right_rows, joint, rest, left_only, right_only, target,
                                event_comparison, left_comparison, right_comparison))
    return tuple(results)
