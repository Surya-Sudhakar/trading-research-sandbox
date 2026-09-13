from __future__ import annotations

"""Reproducible Discovery research pipeline.

One experiment specification + one authorized Discovery partition produces one
immutable result package. The pipeline deliberately does not invent hypotheses;
it executes and records them so Phase 2 can call the same boundary repeatedly.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from sandbox.partition.models import PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.research.context_scan import scan_single_contexts
from sandbox.research.discovery_record_builder import (
    DiscoveryRecordConfig,
    build_discovery_record_population,
)


PIPELINE_VERSION = "SANDBOX_RESEARCH_PIPELINE_V1"
SPEC_VERSION = "SANDBOX_RESEARCH_EXPERIMENT_V1"


@dataclass(frozen=True)
class NumericConditionSpec:
    field_id: str
    cutpoints: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.field_id, str) or not self.field_id.strip():
            raise ValueError("numeric condition field_id must be nonempty")
        cuts = tuple(float(value) for value in self.cutpoints)
        if not cuts:
            raise ValueError("numeric condition requires at least one cutpoint")
        if any(a >= b for a, b in zip(cuts, cuts[1:])):
            raise ValueError("numeric cutpoints must be strictly increasing")
        object.__setattr__(self, "cutpoints", cuts)


@dataclass(frozen=True)
class ResearchPipelineSpec:
    experiment_id: str
    hypothesis: str
    partition_id: str
    categorical_field_ids: tuple[str, ...]
    numeric_condition_specs: tuple[NumericConditionSpec, ...]
    target_field_ids: tuple[str, ...]
    swing_left_bars: int = 2
    swing_right_bars: int = 2
    swing_count_window: int = 8
    version: str = SPEC_VERSION
    mode: str = "DISCOVERY"

    def __post_init__(self) -> None:
        if self.version != SPEC_VERSION:
            raise ValueError(f"unsupported experiment spec version: {self.version}")
        if self.mode != "DISCOVERY":
            raise ValueError("research pipeline V1 only permits DISCOVERY mode")
        for name in ("experiment_id", "hypothesis", "partition_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        for name in ("categorical_field_ids", "target_field_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
            object.__setattr__(self, name, values)
        numeric = tuple(self.numeric_condition_specs)
        if len({item.field_id for item in numeric}) != len(numeric):
            raise ValueError("numeric_condition_specs contains duplicate field IDs")
        object.__setattr__(self, "numeric_condition_specs", numeric)
        if not self.categorical_field_ids and not numeric:
            raise ValueError("at least one research condition is required")
        if not self.target_field_ids:
            raise ValueError("at least one target field is required")
        for name in ("swing_left_bars", "swing_right_bars", "swing_count_window"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchPipelineSpec":
        if not isinstance(payload, dict):
            raise TypeError("experiment spec must be a JSON object")
        allowed = {
            "version", "experiment_id", "hypothesis", "partition_id", "mode",
            "categorical_field_ids", "numeric_condition_specs", "target_field_ids",
            "swing_left_bars", "swing_right_bars", "swing_count_window",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown experiment fields: {', '.join(unknown)}")
        numeric = tuple(
            NumericConditionSpec(item["field_id"], tuple(item["cutpoints"]))
            for item in payload.get("numeric_condition_specs", ())
        )
        return cls(
            version=payload.get("version", SPEC_VERSION),
            experiment_id=payload["experiment_id"],
            hypothesis=payload["hypothesis"],
            partition_id=payload["partition_id"],
            mode=payload.get("mode", "DISCOVERY"),
            categorical_field_ids=tuple(payload.get("categorical_field_ids", ())),
            numeric_condition_specs=numeric,
            target_field_ids=tuple(payload.get("target_field_ids", ())),
            swing_left_bars=payload.get("swing_left_bars", 2),
            swing_right_bars=payload.get("swing_right_bars", 2),
            swing_count_window=payload.get("swing_count_window", 8),
        )


def load_spec(path: Path) -> ResearchPipelineSpec:
    return ResearchPipelineSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _git_commit() -> str:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _result_rows(scans) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in scans:
        comparison = result.comparison
        rows.append({
            "context_kind": result.context_kind,
            "condition_field_id": result.condition_field_id,
            "categorical_value": result.categorical_value,
            "numeric_bin_index": result.numeric_bin_index,
            "numeric_lower_bound": result.numeric_lower_bound,
            "numeric_upper_bound": result.numeric_upper_bound,
            "event_cohort_id": result.event_cohort_id,
            "anchor_kind": result.anchor_kind,
            "pattern_code": result.pattern_code,
            "checkpoint_bars": result.checkpoint_bars,
            "target_field_id": result.target_field_id,
            "subgroup_count": comparison.left_row_count,
            "complement_count": comparison.right_row_count,
            "subgroup_mean": comparison.left_mean,
            "complement_mean": comparison.right_mean,
            "mean_difference": comparison.mean_difference,
            "probability_of_superiority": comparison.probability_of_superiority,
            "cliffs_delta": comparison.cliffs_delta,
        })
    return rows


def run_research_pipeline(
    service: PartitionService,
    spec: ResearchPipelineSpec,
    *,
    results_root: Path = Path("results/research_pipeline"),
) -> dict[str, Any]:
    """Execute one Discovery experiment and persist a reproducible result package."""
    stages: list[dict[str, str]] = []

    manifest = service.get_partition(spec.partition_id)
    if manifest.role != PartitionRole.DISCOVERY:
        raise ValueError("research pipeline V1 can only read DISCOVERY partitions")
    stages.append({"stage": "PARTITION_VERIFY", "status": "SUCCESS"})

    record_config = DiscoveryRecordConfig(
        swing_left_bars=spec.swing_left_bars,
        swing_right_bars=spec.swing_right_bars,
        swing_count_window=spec.swing_count_window,
    )
    population = build_discovery_record_population(service, spec.partition_id, record_config)
    if population.partition_fingerprint != manifest.partition_fingerprint:
        raise ValueError("partition changed during research pipeline execution")
    stages.append({"stage": "FEATURE_BUILD", "status": "SUCCESS"})

    numeric = tuple((item.field_id, item.cutpoints) for item in spec.numeric_condition_specs)
    scans = scan_single_contexts(
        population.dataset,
        categorical_field_ids=spec.categorical_field_ids,
        numeric_condition_specs=numeric,
        field_ids=spec.target_field_ids,
    )
    stages.append({"stage": "EVENT_STUDY", "status": "SUCCESS"})

    spec_payload = asdict(spec)
    git_commit = _git_commit()
    run_identity = {
        "pipeline_version": PIPELINE_VERSION,
        "spec": spec_payload,
        "partition_fingerprint": manifest.partition_fingerprint,
        "source_dataset_checksum": manifest.source_dataset_checksum,
        "git_commit": git_commit,
    }
    run_id = "RUN-" + sha256_canonical(run_identity)[:20].upper()
    result_dir = Path(results_root) / spec.experiment_id / run_id
    result_dir.mkdir(parents=True, exist_ok=False)

    rows = _result_rows(scans)
    metrics = {
        "pipeline_status": "SUCCESS",
        "research_status": "OBSERVED",
        "population_records": len(population.records),
        "event_study_rows": len(rows),
        "categorical_condition_count": len(spec.categorical_field_ids),
        "numeric_condition_count": len(spec.numeric_condition_specs),
        "target_count": len(spec.target_field_ids),
    }
    stages.append({"stage": "METRICS", "status": "SUCCESS"})

    result_payload = {
        "run_id": run_id,
        "experiment_id": spec.experiment_id,
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": git_commit,
        "partition_id": manifest.partition_id,
        "partition_fingerprint": manifest.partition_fingerprint,
        "source_dataset_id": manifest.source_dataset_id,
        "source_dataset_checksum": manifest.source_dataset_checksum,
        "symbol": manifest.symbol,
        "timeframe": manifest.timeframe,
        "start_timestamp": manifest.start_timestamp,
        "end_timestamp": manifest.end_timestamp,
        "row_count": manifest.row_count,
        "spec_fingerprint": sha256_canonical(spec_payload),
        "stages": stages,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    _write_json(result_dir / "experiment.json", spec_payload)
    _write_json(result_dir / "manifest.json", result_payload)
    _write_json(result_dir / "metrics.json", metrics)
    _write_json(result_dir / "event_study.json", rows)
    report = (
        f"# Research run {run_id}\n\n"
        f"**Experiment:** {spec.experiment_id}\n\n"
        f"**Hypothesis:** {spec.hypothesis}\n\n"
        f"**Partition:** {manifest.partition_id} ({manifest.symbol} {manifest.timeframe})\n\n"
        f"**Population records:** {len(population.records)}\n\n"
        f"**Event-study rows:** {len(rows)}\n\n"
        "Pipeline execution completed successfully. `OBSERVED` is not a trading verdict; "
        "Phase 2/3 screening and validation must decide whether an observed effect survives.\n"
    )
    (result_dir / "report.md").write_text(report, encoding="utf-8")

    return {
        "run_id": run_id,
        "result_dir": str(result_dir),
        "manifest": result_payload,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    from sandbox.config import Settings
    from sandbox.research.registry import ResearchRegistry

    parser = argparse.ArgumentParser(description="Run one reproducible Sandbox Discovery experiment")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/research_pipeline"))
    args = parser.parse_args(argv)

    settings = Settings.load()
    registry = ResearchRegistry(settings.catalog_path, Path.cwd(), Path("results"))
    service = PartitionService(registry)
    result = run_research_pipeline(service, load_spec(args.spec), results_root=args.results_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
