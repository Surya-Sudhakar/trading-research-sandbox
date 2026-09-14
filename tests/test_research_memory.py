from dataclasses import replace

import pytest

from sandbox.research.research_memory import ResearchMemory
from sandbox.research.research_pipeline import run_research_pipeline
from test_discovery_record_builder import partition
from test_research_pipeline import spec, varied_context_frame


def candidate(memory, candidate_id, fingerprint, *, parent=None, verdict="REJECTED", reason="REJECT_NEGATIVE_EXPECTANCY"):
    return memory.remember_candidate(
        candidate_id=candidate_id,
        candidate_fingerprint=fingerprint,
        hypothesis_id="H001",
        parent_candidate_id=parent,
        experiment_id="H001-E001",
        run_id="RUN-1",
        parameters={"pivot": 3, "threshold": 0.5},
        metrics={"sample_count": 120, "expectancy": -0.1},
        verdict=verdict,
        rejection_reason=reason if verdict == "REJECTED" else None,
    )


def test_pipeline_persists_run_to_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("sandbox.research.research_pipeline._git_commit", lambda: "abc123")
    service, manifest = partition(tmp_path / "partition", varied_context_frame())
    memory = ResearchMemory(tmp_path / "research.duckdb")
    result = run_research_pipeline(
        service,
        spec(manifest.partition_id),
        results_root=tmp_path / "results",
        memory=memory,
    )
    assert result["memory"]["inserted"] is True
    assert memory.counts() == {"pipeline_runs": 1, "candidates": 0, "rejected_candidates": 0}


def test_pipeline_run_memory_is_idempotent_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr("sandbox.research.research_pipeline._git_commit", lambda: "abc123")
    service, manifest = partition(tmp_path / "partition", varied_context_frame())
    memory = ResearchMemory(tmp_path / "research.duckdb")
    first = run_research_pipeline(service, spec(manifest.partition_id), results_root=tmp_path / "a", memory=memory)
    replay = memory.remember_pipeline_run(
        manifest=first["manifest"], metrics=first["metrics"], hypothesis=spec(manifest.partition_id).hypothesis
    )
    assert replay.inserted is False
    changed_metrics = dict(first["metrics"], event_study_rows=999999)
    with pytest.raises(ValueError, match="immutable research run conflict"):
        memory.remember_pipeline_run(
            manifest=first["manifest"], metrics=changed_metrics, hypothesis=spec(manifest.partition_id).hypothesis
        )


def test_candidate_duplicate_prevention_and_immutability(tmp_path):
    memory = ResearchMemory(tmp_path / "research.duckdb")
    first = candidate(memory, "C-001", "FP-A")
    assert first.inserted is True
    replay = candidate(memory, "C-001", "FP-A")
    assert replay.inserted is False
    assert memory.candidate_exists("FP-A")
    with pytest.raises(ValueError, match="immutable candidate memory conflict"):
        memory.remember_candidate(
            candidate_id="C-001", candidate_fingerprint="FP-A", hypothesis_id="H001",
            experiment_id="H001-E001", run_id="RUN-1", parameters={"pivot": 4}, metrics={},
            verdict="REJECTED", rejection_reason="REJECT_PARAMETER_SENSITIVE"
        )
    with pytest.raises(ValueError, match="duplicate candidate fingerprint"):
        candidate(memory, "C-002", "FP-A")


def test_candidate_lineage_and_failure_summary(tmp_path):
    memory = ResearchMemory(tmp_path / "research.duckdb")
    candidate(memory, "C-001", "FP-1", reason="REJECT_NEGATIVE_EXPECTANCY")
    candidate(memory, "C-002", "FP-2", parent="C-001", reason="REJECT_PARAMETER_SENSITIVE")
    candidate(memory, "C-003", "FP-3", parent="C-001", reason="REJECT_PARAMETER_SENSITIVE")
    candidate(memory, "C-004", "FP-4", parent="C-002", verdict="PROMISING")
    assert [row["candidate_id"] for row in memory.children_of("C-001")] == ["C-002", "C-003"]
    assert memory.failure_summary() == (
        {"rejection_reason": "REJECT_PARAMETER_SENSITIVE", "candidate_count": 2},
        {"rejection_reason": "REJECT_NEGATIVE_EXPECTANCY", "candidate_count": 1},
    )
    assert memory.counts() == {"pipeline_runs": 0, "candidates": 4, "rejected_candidates": 3}


def test_rejected_candidate_requires_reason(tmp_path):
    memory = ResearchMemory(tmp_path / "research.duckdb")
    with pytest.raises(ValueError, match="requires rejection_reason"):
        memory.remember_candidate(
            candidate_id="C-001", candidate_fingerprint="FP-1", experiment_id="E-1", run_id="R-1",
            parameters={}, metrics={}, verdict="REJECTED"
        )
