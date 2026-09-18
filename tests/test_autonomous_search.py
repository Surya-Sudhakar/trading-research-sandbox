from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path

import pytest

from sandbox.research.autonomous_search import (
    CandidateCondition, CandidateSpec, NumericOperator, RejectionReason,
    SearchField, SearchFieldKind, SearchSpaceSpec, candidate_fingerprint_payload,
    filter_unseen_candidates, generate_candidates, load_search_space_spec,
)
from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.research.research_memory import ResearchMemory


def search_spec(max_conditions=2):
    return SearchSpaceSpec(
        search_id="SEARCH_V1", experiment_id="EXP-1", hypothesis_id="H-1",
        partition_id="PART-DISCOVERY", target_field_ids=("y.h4.close_return_fraction",),
        fields=(
            SearchField("x.level_transition.sweep", SearchFieldKind.CATEGORICAL,
                        categorical_values=("ABOVE", "BELOW")),
            SearchField("x.daily.previous.level.low.close_distance_atr", SearchFieldKind.NUMERIC,
                        threshold_values=(0, .25), operators=(NumericOperator.LTE, NumericOperator.GT)),
        ),
        swing_left_values=(2, 3), swing_right_values=(2,),
        swing_count_window_values=(6, 8), max_conditions_per_candidate=max_conditions,
    )


def test_deterministic_fingerprint_and_candidate_id():
    first = generate_candidates(search_spec())[0]
    second = generate_candidates(search_spec())[0]
    assert first == second
    assert first.candidate_fingerprint == second.candidate_fingerprint
    assert first.candidate_id == "CAND-" + first.candidate_fingerprint[:20].upper()
    assert len(first.candidate_fingerprint) == 64


def test_dictionary_condition_and_target_order_do_not_change_identity():
    conditions = (
        CandidateCondition("x.level_transition.sweep", "=", "BELOW"),
        CandidateCondition("x.daily.previous.level.low.close_distance_atr", "<=", .25),
    )
    arguments = dict(search_id="S", experiment_id="E", partition_id="P",
                     conditions=conditions, target_field_ids=("y.h8.close_return_fraction",
                     "y.h4.close_return_fraction"), swing_left_bars=2,
                     swing_right_bars=3, swing_count_window=8)
    first = candidate_fingerprint_payload(**arguments)
    second = candidate_fingerprint_payload(**{
        **arguments, "conditions": conditions[::-1],
        "target_field_ids": arguments["target_field_ids"][::-1],
    })
    assert sha256_canonical(first) == sha256_canonical(dict(reversed(tuple(second.items()))))


def test_deterministic_order_and_byte_equivalent_replay():
    first = generate_candidates(search_spec())
    second = generate_candidates(search_spec())
    assert first == second
    assert canonical_json([asdict(candidate) for candidate in first]) == canonical_json(
        [asdict(candidate) for candidate in second])
    assert len(first) == 84
    assert first[0].conditions == (CandidateCondition("x.level_transition.sweep", "=", "ABOVE"),)
    assert [(candidate.swing_left_bars, candidate.swing_count_window)
            for candidate in first[:4]] == [(2, 6), (2, 8), (3, 6), (3, 8)]


def test_maximum_condition_count_and_empty_candidate_impossible():
    one = generate_candidates(search_spec(max_conditions=1))
    two = generate_candidates(search_spec(max_conditions=2))
    assert one and two
    assert all(len(candidate.conditions) == 1 for candidate in one)
    assert max(len(candidate.conditions) for candidate in two) == 2
    with pytest.raises(ValueError, match="positive integer"):
        replace(search_spec(), max_conditions_per_candidate=0)
    with pytest.raises(ValueError, match="nonempty"):
        replace(two[0], conditions=())


def test_numeric_operators_and_thresholds_are_explicit():
    candidates = generate_candidates(search_spec(max_conditions=1))
    numeric = {(condition.operator, condition.value) for candidate in candidates
               for condition in candidate.conditions if condition.field_id.endswith("distance_atr")}
    assert numeric == {("<=", 0.0), ("<=", .25), (">", 0.0), (">", .25)}


def test_default_numeric_operators_cover_required_relations():
    field = SearchField("x.market.m15.volatility.atr", "numeric", threshold_values=(1,))
    assert tuple(operator.value for operator in field.operators) == ("<", "<=", ">", ">=")


def test_categorical_conditions_preserve_typed_values():
    field = SearchField("x.level_transition.sweep", "categorical",
                        categorical_values=("BELOW", False, 0))
    spec = replace(search_spec(max_conditions=1), fields=(field,))
    values = [candidate.conditions[0].value for candidate in generate_candidates(spec)[::4]]
    assert values == ["BELOW", False, 0]


def test_pivot_parameter_cartesian_variation():
    candidates = generate_candidates(search_spec(max_conditions=1))
    observed = {(candidate.swing_left_bars, candidate.swing_right_bars,
                 candidate.swing_count_window) for candidate in candidates}
    assert observed == {(2, 2, 6), (2, 2, 8), (3, 2, 6), (3, 2, 8)}


@pytest.mark.parametrize("field", [
    SearchField("x.valid.field", "categorical", categorical_values=("A",)),
    SearchField("x.valid.numeric", "numeric", threshold_values=(1,)),
])
def test_models_are_frozen(field):
    with pytest.raises(FrozenInstanceError):
        field.field_id = "x.other.field"
    candidate = generate_candidates(search_spec())[0]
    with pytest.raises(FrozenInstanceError):
        candidate.swing_left_bars = 99


@pytest.mark.parametrize("factory", [
    lambda: SearchField("", "categorical", categorical_values=("A",)),
    lambda: SearchField("y.target.value", "categorical", categorical_values=("A",)),
    lambda: SearchField("x.valid.field", "categorical"),
    lambda: SearchField("x.valid.field", "numeric", threshold_values=()),
    lambda: SearchField("x.valid.field", "numeric", threshold_values=(float("nan"),)),
    lambda: SearchField("x.valid.field", "categorical", categorical_values=("A", "A")),
    lambda: replace(search_spec(), fields=()),
    lambda: replace(search_spec(), swing_left_values=(0,)),
    lambda: replace(search_spec(), target_field_ids=()),
])
def test_invalid_or_empty_fields_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_unknown_json_fields_rejected_at_top_and_nested(tmp_path):
    payload = {
        "search_id": "S", "experiment_id": "E", "partition_id": "P",
        "target_field_ids": ["y.h4.close_return_fraction"],
        "fields": [{"field_id": "x.level_transition.sweep", "kind": "categorical",
                    "categorical_values": ["BELOW"]}],
        "swing_left_values": [2], "swing_right_values": [2],
        "swing_count_window_values": [8], "max_conditions_per_candidate": 1,
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps({**payload, "unknown": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown search specification"):
        load_search_space_spec(path)
    payload["fields"][0]["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown search field"):
        load_search_space_spec(path)


def test_duplicate_condition_prevention():
    with pytest.raises(ValueError, match="duplicates"):
        SearchField("x.level_transition.sweep", "categorical",
                    categorical_values=("BELOW", "BELOW"))
    candidate = generate_candidates(search_spec())[0]
    with pytest.raises(ValueError, match="duplicate conditions"):
        CandidateSpec(
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.candidate_fingerprint,
            search_id=candidate.search_id, experiment_id=candidate.experiment_id,
            hypothesis_id=candidate.hypothesis_id, partition_id=candidate.partition_id,
            conditions=(candidate.conditions[0], candidate.conditions[0]),
            target_field_ids=candidate.target_field_ids,
            swing_left_bars=candidate.swing_left_bars,
            swing_right_bars=candidate.swing_right_bars,
            swing_count_window=candidate.swing_count_window,
        )


def test_duplicate_filter_calls_existing_research_memory(tmp_path):
    candidates = generate_candidates(search_spec(max_conditions=1))
    memory = ResearchMemory(tmp_path / "research.duckdb")
    seen = candidates[1]
    memory.remember_candidate(
        candidate_id=seen.candidate_id, candidate_fingerprint=seen.candidate_fingerprint,
        hypothesis_id=seen.hypothesis_id, experiment_id=seen.experiment_id,
        run_id="RUN-1", parameters={}, metrics={}, verdict="PROMISING",
    )
    unseen = filter_unseen_candidates(candidates, memory)
    assert seen not in unseen
    assert unseen == candidates[:1] + candidates[2:]
    assert seen.candidate_fingerprint == candidates[1].candidate_fingerprint


def test_identity_has_no_runtime_random_or_lineage_values():
    candidate = generate_candidates(search_spec())[0]
    payload = candidate_fingerprint_payload(
        search_id=candidate.search_id, experiment_id=candidate.experiment_id,
        partition_id=candidate.partition_id, conditions=candidate.conditions,
        target_field_ids=candidate.target_field_ids,
        swing_left_bars=candidate.swing_left_bars,
        swing_right_bars=candidate.swing_right_bars,
        swing_count_window=candidate.swing_count_window,
    )
    encoded = canonical_json(payload)
    assert set(payload) == {
        "version", "search_id", "experiment_id", "partition_id", "conditions",
        "target_field_ids", "swing_left_bars", "swing_right_bars", "swing_count_window",
    }
    assert "timestamp" not in encoded and "uuid" not in encoded and "parent_candidate_id" not in encoded
    changed = replace(candidate, hypothesis_id="OTHER", parent_candidate_id="PARENT")
    assert changed != candidate
    assert changed.candidate_id == candidate.candidate_id
    assert changed.candidate_fingerprint == candidate.candidate_fingerprint


def test_example_json_loads_and_uses_current_discovery_fields():
    spec = load_search_space_spec(Path("experiments/search_previous_day_structure_v1.json"))
    assert spec.search_id == "PREVIOUS_DAY_STRUCTURE_V1"
    assert {field.field_id for field in spec.fields} == {
        "x.level_transition.sweep", "x.level_transition.reclaim",
        "x.market.m15.swing.high.label", "x.market.m15.swing.low.label",
        "x.daily.previous.level.high.close_distance_atr",
        "x.daily.previous.level.low.close_distance_atr",
    }
    assert all(target.startswith("y.") for target in spec.target_field_ids)


def test_rejection_taxonomy_is_frozen():
    assert {reason.value for reason in RejectionReason} == {
        "REJECT_LOW_SAMPLE", "REJECT_NO_EFFECT", "REJECT_NEGATIVE_EFFECT",
        "REJECT_UNSTABLE_ACROSS_YEARS", "REJECT_PARAMETER_SENSITIVE",
        "REJECT_SESSION_DEPENDENT", "REJECT_SPREAD_SENSITIVE", "REJECT_OOS_FAILURE",
        "REJECT_DUPLICATE_HYPOTHESIS",
    }


def test_search_accepts_existing_single_component_predictor_ids():
    field = SearchField("x.anchor_kind", "categorical", categorical_values=("TOUCH_TRANSITION",))
    spec = replace(search_spec(max_conditions=1), fields=(field,))
    assert generate_candidates(spec)[0].conditions[0].field_id == "x.anchor_kind"
