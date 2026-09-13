from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from sandbox import RESEARCH_SCHEMA_VERSION


def _normalize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite numbers are not canonical")
        return json.loads(json.dumps(value))
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def experiment_fingerprint(*, question_id: str, hypothesis_id: str, dataset_ids, strategy_spec, execution_config, evaluation_spec, code_version, schema_version: str = RESEARCH_SCHEMA_VERSION) -> str:
    return sha256_canonical({"question_id": question_id, "hypothesis_id": hypothesis_id, "dataset_ids": sorted(dataset_ids), "strategy_spec": strategy_spec, "execution_config": execution_config, "evaluation_spec": evaluation_spec, "code_version": code_version, "research_schema_version": schema_version})

