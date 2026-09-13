"""Observed single-context subgroups versus their same-event complements."""
from dataclasses import dataclass
from math import isfinite
from sandbox.market_state.discovery_dataset import DiscoveryDataset, field_role
from .conditional_event_summary import summarize_single_conditioned_event_cohorts
from .numeric_condition_summary import NumericConditionalEventCohortSummary, summarize_numeric_conditioned_event_cohorts
from .cohort_comparison import NumericCohortComparison, compare_cohort_rows


@dataclass(frozen=True)
class SingleContextScanResult:
    context_kind: str
    condition_field_id: str
    numeric_cutpoints: tuple[float, ...] | None
    event_cohort_id: str
    anchor_kind: str
    pattern_code: str | None
    checkpoint_bars: int | None
    subgroup_row_indices: tuple[int, ...]
    complement_row_indices: tuple[int, ...]
    target_field_id: str
    comparison: NumericCohortComparison
    categorical_value: None | bool | int | str
    numeric_missing: bool | None
    numeric_bin_index: int | None
    numeric_lower_bound: float | None
    numeric_upper_bound: float | None
    numeric_lower_inclusive: bool | None
    numeric_upper_inclusive: bool | None

    def __post_init__(self):
        if type(self.context_kind) is not str or self.context_kind not in ("CATEGORICAL", "NUMERIC"):
            raise ValueError("unsupported context kind")
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.condition_field_id, self.event_cohort_id, self.anchor_kind, self.target_field_id)):
            raise ValueError("identity fields must be nonempty strings")
        for indices in (self.subgroup_row_indices, self.complement_row_indices):
            if not isinstance(indices, tuple) or not indices or any(type(i) is not int or i < 0 for i in indices):
                raise ValueError("indices must be nonempty tuples of nonnegative integers")
            if any(a >= b for a, b in zip(indices, indices[1:])):
                raise ValueError("indices must be strictly increasing")
        if set(self.subgroup_row_indices).intersection(self.complement_row_indices):
            raise ValueError("subgroup and complement must be disjoint")
        comparison = self.comparison
        if not isinstance(comparison, NumericCohortComparison) or (
                comparison.field_id != self.target_field_id or
                comparison.left_row_indices != self.subgroup_row_indices or
                comparison.right_row_indices != self.complement_row_indices):
            raise ValueError("comparison does not match scan trace")
        metadata = (self.numeric_missing, self.numeric_bin_index, self.numeric_lower_bound,
                    self.numeric_upper_bound, self.numeric_lower_inclusive, self.numeric_upper_inclusive)
        if self.context_kind == "CATEGORICAL":
            if self.numeric_cutpoints is not None or any(v is not None for v in metadata):
                raise ValueError("categorical contexts cannot have numeric metadata")
            if self.categorical_value is not None and type(self.categorical_value) not in (bool, int, str):
                raise ValueError("unsupported categorical value")
            return
        cuts = self.numeric_cutpoints
        if not isinstance(cuts, tuple) or not cuts or any(type(c) is not float or not isfinite(c) for c in cuts):
            raise ValueError("numeric cutpoints must be a nonempty tuple of finite floats")
        if any(a >= b for a, b in zip(cuts, cuts[1:])):
            raise ValueError("numeric cutpoints must be strictly increasing")
        if self.categorical_value is not None:
            raise ValueError("numeric contexts cannot have categorical values")
        # Reuse the established interval invariants without computing outcomes.
        NumericConditionalEventCohortSummary(self.event_cohort_id, self.anchor_kind,
            self.pattern_code, self.checkpoint_bars, self.condition_field_id, *metadata,
            self.subgroup_row_indices, len(self.subgroup_row_indices), ())
        if not self.numeric_missing:
            index = self.numeric_bin_index
            if index > len(cuts):
                raise ValueError("bin index exceeds explicit cutpoints")
            lower = cuts[index - 1] if index else None
            upper = cuts[index] if index < len(cuts) else None
            if self.numeric_lower_bound != lower or self.numeric_upper_bound != upper:
                raise ValueError("interval does not match explicit cutpoints")


def scan_single_contexts(dataset: DiscoveryDataset, categorical_field_ids=(),
                         numeric_condition_specs=(), field_ids=None) -> tuple[SingleContextScanResult, ...]:
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not dataset.rows:
        return ()
    categorical = tuple(categorical_field_ids)
    if any(not isinstance(f, str) for f in categorical):
        raise TypeError("categorical field IDs must be strings")
    if len(set(categorical)) != len(categorical):
        raise ValueError("duplicate categorical field IDs")
    numeric = tuple(numeric_condition_specs)
    fields = dataset.target_field_ids if field_ids is None else tuple(field_ids)
    if any(not isinstance(f, str) for f in fields):
        raise TypeError("target field IDs must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate target field IDs")
    for field in fields:
        if field_role(dataset, field) != "target":
            raise ValueError("scan targets must be target fields")
    results = []

    def append_scan(kind, field, cuts, cohorts):
        events = {}
        for cohort in cohorts:
            events.setdefault(cohort.event_cohort_id, []).append(cohort)
        for event_id in sorted(events):
            groups = events[event_id]
            event_rows = sorted({i for group in groups for i in group.row_indices})
            for group in groups:
                members = set(group.row_indices)
                complement = tuple(i for i in event_rows if i not in members)
                if not complement:
                    continue
                metadata = (group.missing, group.bin_index, group.lower_bound, group.upper_bound,
                            group.lower_inclusive, group.upper_inclusive) if kind == "NUMERIC" else (None,) * 6
                value = group.condition_values[0] if kind == "CATEGORICAL" else None
                for target in fields:
                    comparison = compare_cohort_rows(dataset, group.row_indices, complement, target)
                    results.append(SingleContextScanResult(kind, field, cuts, event_id,
                        group.anchor_kind, group.pattern_code, group.checkpoint_bars,
                        group.row_indices, complement, target, comparison, value, *metadata))

    for field in categorical:
        cohorts = summarize_single_conditioned_event_cohorts(dataset, field, field_ids=())
        append_scan("CATEGORICAL", field, None, cohorts)
    for field, cutpoints in numeric:
        cuts = tuple(cutpoints)
        cohorts = summarize_numeric_conditioned_event_cohorts(dataset, field, cuts, field_ids=())
        # The numeric engine validates cuts before their normalized identity is retained.
        normalized = tuple(float(c) for c in cuts)
        append_scan("NUMERIC", field, normalized, cohorts)
    return tuple(results)
