"""Caller-controlled, direction-neutral ordering of discovery evidence."""
from dataclasses import dataclass
from math import isfinite

from .context_scan import SingleContextScanResult
from .context_interaction_scan import TwoContextScanResult


def _comparisons(result):
    if isinstance(result, SingleContextScanResult):
        return (result.comparison,)
    return tuple(c for c in (result.joint_vs_event_complement,
        result.joint_vs_left_without_right, result.joint_vs_right_without_left) if c is not None)


@dataclass(frozen=True)
class RankedContextCandidate:
    candidate_kind: str
    target_field_id: str
    event_cohort_id: str
    priority_score: float
    minimum_comparison_pair_count: int
    minimum_comparison_observed_per_side: int
    source_order: int
    single_result: SingleContextScanResult | None
    interaction_result: TwoContextScanResult | None

    def __post_init__(self):
        if type(self.candidate_kind) is not str or self.candidate_kind not in ("SINGLE", "INTERACTION"):
            raise ValueError("unsupported candidate kind")
        if self.candidate_kind == "SINGLE":
            if not isinstance(self.single_result, SingleContextScanResult) or self.interaction_result is not None:
                raise ValueError("single candidate requires only a single result")
            source = self.single_result
        else:
            if not isinstance(self.interaction_result, TwoContextScanResult) or self.single_result is not None:
                raise ValueError("interaction candidate requires only an interaction result")
            source = self.interaction_result
        if self.target_field_id != source.target_field_id or self.event_cohort_id != source.event_cohort_id:
            raise ValueError("candidate identity must match source")
        if type(self.source_order) is not int or self.source_order < 0:
            raise ValueError("source order must be nonnegative integer")
        for value in (self.minimum_comparison_pair_count, self.minimum_comparison_observed_per_side):
            if type(value) is not int or value < 1:
                raise ValueError("minimum counts must be positive integers")
        if type(self.priority_score) is not float or not isfinite(self.priority_score) or not 0 <= self.priority_score <= 1:
            raise ValueError("priority must be finite float in [0, 1]")
        comparisons = _comparisons(source)
        if any(c.cliffs_delta is None for c in comparisons):
            raise ValueError("candidate comparisons must have observed pairs")
        if (self.priority_score != min(abs(c.cliffs_delta) for c in comparisons) or
            self.minimum_comparison_pair_count != min(c.pair_count for c in comparisons) or
            self.minimum_comparison_observed_per_side != min(min(c.left_observed_count, c.right_observed_count) for c in comparisons)):
            raise ValueError("candidate metrics must match source comparisons")


def rank_context_candidates(single_results=(), interaction_results=(), *, minimum_rows_per_side=1,
                            minimum_observed_per_side=1, minimum_pair_count=1,
                            minimum_abs_cliffs_delta=0.0, require_incremental_baselines=True,
                            top_k=None) -> tuple[RankedContextCandidate, ...]:
    for value in (minimum_rows_per_side, minimum_observed_per_side, minimum_pair_count):
        if type(value) is not int or value < 1:
            raise ValueError("minimum counts must be positive integers")
    if type(minimum_abs_cliffs_delta) not in (int, float):
        raise ValueError("minimum delta must be numeric")
    try:
        minimum_delta = float(minimum_abs_cliffs_delta)
    except OverflowError as exc:
        raise ValueError("minimum delta must be finite") from exc
    if not isfinite(minimum_delta) or not 0 <= minimum_delta <= 1:
        raise ValueError("minimum delta must be in [0, 1]")
    if type(require_incremental_baselines) is not bool:
        raise ValueError("incremental baseline control must be bool")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be positive integer or None")
    singles, interactions = tuple(single_results), tuple(interaction_results)
    if any(not isinstance(r, SingleContextScanResult) for r in singles) or any(not isinstance(r, TwoContextScanResult) for r in interactions):
        raise TypeError("inputs must contain their declared scan result types")
    candidates = []
    for order, result in enumerate(singles + interactions):
        single = order < len(singles)
        if not single and require_incremental_baselines and (
            result.joint_vs_left_without_right is None or result.joint_vs_right_without_left is None):
            continue
        comparisons = _comparisons(result)
        if any(c.left_row_count < minimum_rows_per_side or c.right_row_count < minimum_rows_per_side or
               c.left_observed_count < minimum_observed_per_side or c.right_observed_count < minimum_observed_per_side or
               c.pair_count < minimum_pair_count or c.cliffs_delta is None for c in comparisons):
            continue
        score = min(abs(c.cliffs_delta) for c in comparisons)
        if score < minimum_delta:
            continue
        candidates.append(RankedContextCandidate("SINGLE" if single else "INTERACTION",
            result.target_field_id, result.event_cohort_id, score,
            min(c.pair_count for c in comparisons),
            min(min(c.left_observed_count, c.right_observed_count) for c in comparisons), order,
            result if single else None, None if single else result))
    candidates.sort(key=lambda c: (-c.priority_score, -c.minimum_comparison_pair_count,
        -c.minimum_comparison_observed_per_side, c.event_cohort_id, c.target_field_id,
        c.candidate_kind, c.source_order))
    return tuple(candidates if top_k is None else candidates[:top_k])
