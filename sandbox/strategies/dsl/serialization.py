"""Versioned canonical serialization of a complete resolved StrategySpec."""
import hashlib
import json

from .specification import StrategySpec

STRATEGY_SPEC_CANONICAL_VERSION = "STRATEGY_SPEC_V1"


def canonical_strategy_spec_json(spec: StrategySpec) -> str:
    """Revalidate the full tree and preserve all rule, child, and value ordering."""
    validated = StrategySpec.model_validate(spec)
    envelope = {
        "schema_version": STRATEGY_SPEC_CANONICAL_VERSION,
        "spec": validated.model_dump(mode="json"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def strategy_spec_fingerprint(spec: StrategySpec) -> str:
    """SHA-256 of canonical UTF-8 bytes, without external provenance data."""
    return hashlib.sha256(canonical_strategy_spec_json(spec).encode("utf-8")).hexdigest()
