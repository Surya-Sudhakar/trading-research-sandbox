from dataclasses import replace
import json
from pathlib import Path

import pytest

from sandbox.partition.models import PartitionRole
from sandbox.research.research_pipeline import (
    ResearchPipelineSpec,
    run_research_pipeline,
)
from test_discovery_record_builder import market_frame, partition


def spec(partition_id):
    return ResearchPipelineSpec(
        experiment_id="H001-E001",
        hypothesis="Previous-day context may contain predictive information.",
        partition_id=partition_id,
        categorical_field_ids=("x.m15.direction",),
        numeric_condition_specs=(),
        target_field_ids=("y.h4.close_return_fraction",),
        swing_left_bars=3,
        swing_right_bars=3,
        swing_count_window=8,
    )


def test_pipeline_writes_reproducible_result_package(tmp_path, monkeypatch):
    service, manifest = partition(tmp_path / "partition", market_frame(64, touches=True))
    monkeypatch.setattr("sandbox.research.research_pipeline._git_commit", lambda: "abc123")
    result = run_research_pipeline(service, spec(manifest.partition_id), results_root=tmp_path / "results")
    folder = Path(result["result_dir"])
    assert {p.name for p in folder.iterdir()} == {
        "experiment.json", "manifest.json", "metrics.json", "event_study.json", "report.md"
    }
    stored_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert stored_manifest["git_commit"] == "abc123"
    assert stored_manifest["partition_fingerprint"] == manifest.partition_fingerprint
    assert metrics["pipeline_status"] == "SUCCESS"
    assert metrics["research_status"] == "OBSERVED"
    assert metrics["population_records"] > 0
    assert metrics["event_study_rows"] > 0


def test_run_identity_changes_with_pivot_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr("sandbox.research.research_pipeline._git_commit", lambda: "abc123")
    service, manifest = partition(tmp_path / "partition", market_frame(64, touches=True))
    first = run_research_pipeline(service, spec(manifest.partition_id), results_root=tmp_path / "a")
    second_spec = replace(spec(manifest.partition_id), swing_left_bars=4, swing_right_bars=4)
    second = run_research_pipeline(service, second_spec, results_root=tmp_path / "b")
    assert first["run_id"] != second["run_id"]


def test_pipeline_refuses_protected_partition(tmp_path):
    service, manifest = partition(
        tmp_path / "partition", market_frame(64, touches=True), PartitionRole.VALIDATION
    )
    with pytest.raises(ValueError, match="DISCOVERY"):
        run_research_pipeline(service, spec(manifest.partition_id), results_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_spec_requires_positive_pivot_parameters():
    with pytest.raises(ValueError, match="swing_left_bars"):
        replace(spec("P1"), swing_left_bars=0)
