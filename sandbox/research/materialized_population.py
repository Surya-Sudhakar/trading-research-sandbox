"""One-time compact execution-view materialization directly from ResearchRecords.

This deliberately avoids rebuilding the full DiscoveryDataset (hundreds of
thousands of rows x ~150 scalar wrapper objects). For the frozen production
packet features, values are extracted directly from the same causal record
fields used by discovery_context_projection. Unknown future packet features
fall back to the canonical projection one record at a time, so memory remains
bounded by the compact column arrays rather than a full projected dataset.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from sandbox.market_state.discovery_context_projection import project_research_record_with_context
from sandbox.provenance import sha256_file

from .ai_researcher import AIResearchPacket
from .canonical import canonical_json, sha256_canonical
from .compiled_strategy_execution import _nonblank, _number, _timestamp
from .materialized_execution import (
    MATERIALIZED_EXECUTION_VIEW_VERSION,
    MANIFEST_FILENAME,
    PARQUET_FILENAME,
    _BASE_COLUMNS,
    _field_value,
    _manifest_state,
    _validate_sha,
    load_execution_view,
)


_DIRECT_FEATURES = {
    "x.anchor_kind",
    "x.geometry.coordinate",
    "x.geometry.source_block_label",
    "x.event.pattern_code",
    "x.event.checkpoint_bars",
    "x.m15.body_to_range",
    "x.m15.range_fraction",
}


def _direct_values(record):
    anchor = record.anchor
    reference = _number(anchor.outcome_anchor.reference_price)
    if reference <= 0:
        raise ValueError("anchor reference price must be positive")
    base = record.multitimeframe_context.snapshot.base_bar
    span = base.high - base.low
    return {
        "x.anchor_kind": anchor.kind.value,
        "x.geometry.coordinate": anchor.coordinate,
        "x.geometry.source_block_label": anchor.source_block_label,
        "x.event.pattern_code": anchor.pattern_code,
        "x.event.checkpoint_bars": anchor.checkpoint_bars,
        "x.m15.body_to_range": abs(base.close - base.open) / span if span > 0 else None,
        "x.m15.range_fraction": span / abs(reference) if reference != 0 else None,
    }


def _feature_values(record, feature_ids):
    direct = _direct_values(record)
    unknown = [field for field in feature_ids if field not in _DIRECT_FEATURES]
    if unknown:
        projected = project_research_record_with_context(record)
        canonical = {value.field_id: value.value for value in projected.predictors}
        missing = [field for field in unknown if field not in canonical]
        if missing:
            raise ValueError(f"packet feature missing from canonical projection: {missing[0]}")
        direct.update({field: canonical[field] for field in unknown})
    return direct


def materialize_records_execution_view(
    records,
    packet: AIResearchPacket,
    *,
    directory: Path,
    prepared_id: str,
    dataset_id: str,
    symbol: str,
    timeframe: str,
):
    """Create the compact Parquet view without constructing DiscoveryDataset."""
    if type(packet) is not AIResearchPacket:
        raise TypeError("expected exact AIResearchPacket")
    packet.__post_init__()
    _validate_sha(prepared_id, "prepared_id")
    for value in (dataset_id, symbol, timeframe):
        _nonblank(value)
    records = tuple(records)

    feature_ids = tuple(packet.allowed_condition_feature_ids)
    kinds = dict(zip(packet.feature_ids, packet.feature_kinds))
    feature_kinds = tuple(kinds[field] for field in feature_ids)

    row_indices = []
    timestamps = []
    reference_prices = []
    anchor_kinds = []
    geometry_family_ids = []
    geometry_level_ids = []
    feature_columns = {field: [] for field in feature_ids}

    for row_index, record in enumerate(records):
        anchor = record.anchor
        reference = _number(anchor.outcome_anchor.reference_price)
        if reference <= 0:
            raise ValueError("anchor reference price must be positive")
        values = _feature_values(record, feature_ids)
        row_indices.append(row_index)
        timestamps.append(_timestamp(anchor.evidence_end_utc))
        reference_prices.append(reference)
        anchor_kinds.append(anchor.kind.value)
        geometry_family_ids.append(anchor.geometry_family_id)
        geometry_level_ids.append(anchor.level_id)
        for field in feature_ids:
            feature_columns[field].append(_field_value(values[field], kinds[field]))

    arrays = [
        pa.array(row_indices, type=pa.int64()),
        pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
        pa.array(reference_prices, type=pa.float64()),
        pa.array(anchor_kinds, type=pa.string()),
        pa.array(geometry_family_ids, type=pa.string()),
        pa.array(geometry_level_ids, type=pa.string()),
    ]
    names = list(_BASE_COLUMNS)
    for field, kind in zip(feature_ids, feature_kinds):
        arrays.append(pa.array(feature_columns[field], type=pa.float64() if kind == "NUMERIC" else pa.string()))
        names.append(field)
    table = pa.Table.from_arrays(arrays, names=names)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / PARQUET_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    temporary = directory / f".{PARQUET_FILENAME}.{uuid4().hex}.tmp"
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True, write_statistics=True)
        parquet_sha256 = sha256_file(temporary)
        state = _manifest_state(
            version=MATERIALIZED_EXECUTION_VIEW_VERSION,
            prepared_id=prepared_id,
            packet_fingerprint=packet.fingerprint,
            dataset_id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            row_count=len(records),
            feature_ids=feature_ids,
            feature_kinds=feature_kinds,
            parquet_filename=PARQUET_FILENAME,
            parquet_sha256=parquet_sha256,
        )
        fingerprint = sha256_canonical(state)
        document = {**state, "fingerprint": fingerprint}

        if parquet_path.exists() or manifest_path.exists():
            if not (parquet_path.exists() and manifest_path.exists()):
                raise ValueError("partial materialized execution view exists")
            existing = load_execution_view(directory)
            # Existing immutable output is accepted only when all lineage/schema
            # fields and deterministic Parquet checksum match this regeneration.
            if _manifest_state(existing) != state:
                raise ValueError("existing execution view does not match requested lineage")
            return existing

        temporary.replace(parquet_path)
        manifest_path.write_text(canonical_json(document), encoding="utf-8")
        return load_execution_view(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
