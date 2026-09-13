"""Compact Parquet/DuckDB execution view for repeated Discovery hypotheses.

This layer is derived from an already-prepared DiscoveryDataset. It contains only
causal predictor columns allowed by the AIResearchPacket plus the anchor metadata
required to reproduce TradeIntents. Targets are deliberately excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from uuid import uuid4
import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from sandbox.execution.models import Direction, EntryType, TradeIntent
from sandbox.market_state.discovery_dataset import DiscoveryDataset
from sandbox.provenance import sha256_file
from sandbox.strategies.dsl.conditions import AndCondition, ComparisonCondition
from sandbox.strategies.dsl.execution import MarketEntry, FixedPipsStop, RiskMultipleTarget

from .ai_researcher import AIResearchPacket, _digest, _sha
from .canonical import canonical_json, sha256_canonical
from .compiled_strategy_execution import (
    COMPILED_STRATEGY_EXECUTION_VERSION,
    CompiledStrategyIntentBatch,
    _check_condition,
    _geometry,
    _intent_id,
    _nonblank,
    _number,
    _timestamp,
)
from .hypothesis_compiler import CompiledHypothesis


MATERIALIZED_EXECUTION_VIEW_VERSION = "MATERIALIZED_EXECUTION_VIEW_V1"
PARQUET_FILENAME = "execution_view.parquet"
MANIFEST_FILENAME = "execution_view.json"
_BASE_COLUMNS = (
    "source_row_index",
    "evidence_end_utc",
    "reference_price",
    "anchor_kind",
    "geometry_family_id",
    "geometry_level_id",
)


def _finite_number(value):
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("numeric execution-view value must be int/float excluding bool")
    value = float(value)
    if not isfinite(value):
        raise ValueError("numeric execution-view value must be finite")
    return value


def _categorical(value):
    """Encode categorical scalars while preserving Python type identity."""
    if value is None:
        return None
    if type(value) is bool:
        payload = ("bool", value)
    elif type(value) is int:
        payload = ("int", str(value))
    elif type(value) is float:
        if not isfinite(value):
            raise ValueError("categorical float must be finite")
        # Python equality treats -0.0 == 0.0; normalize that representation.
        payload = ("float", "0.0" if value == 0.0 else repr(value))
    elif type(value) is str:
        payload = ("str", value)
    else:
        raise ValueError("unsupported categorical execution-view value")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _field_value(value, kind):
    if kind == "NUMERIC":
        return _finite_number(value)
    if kind == "CATEGORICAL":
        return _categorical(value)
    raise ValueError("unsupported feature kind")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _validate_sha(value: str, label: str) -> None:
    try:
        _sha(value)
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc


@dataclass(frozen=True)
class MaterializedExecutionView:
    version: str
    prepared_id: str
    packet_fingerprint: str
    dataset_id: str
    symbol: str
    timeframe: str
    row_count: int
    feature_ids: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    parquet_path: Path
    parquet_sha256: str
    fingerprint: str

    def __post_init__(self):
        if self.version != MATERIALIZED_EXECUTION_VIEW_VERSION:
            raise ValueError("unsupported materialized execution-view version")
        _validate_sha(self.prepared_id, "prepared_id")
        _validate_sha(self.packet_fingerprint, "packet_fingerprint")
        _validate_sha(self.parquet_sha256, "parquet_sha256")
        _validate_sha(self.fingerprint, "fingerprint")
        for value in (self.dataset_id, self.symbol, self.timeframe):
            _nonblank(value)
        if type(self.row_count) is not int or self.row_count < 0:
            raise ValueError("row_count must be a nonnegative exact integer")
        if type(self.feature_ids) is not tuple or type(self.feature_kinds) is not tuple:
            raise ValueError("feature schema must be tuples")
        if not self.feature_ids or len(self.feature_ids) != len(self.feature_kinds):
            raise ValueError("feature IDs/kinds must be nonempty and aligned")
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("duplicate feature IDs")
        if any(type(v) is not str or not v.startswith("x.") for v in self.feature_ids):
            raise ValueError("execution-view features must be x.* predictors")
        if any(k not in ("NUMERIC", "CATEGORICAL") for k in self.feature_kinds):
            raise ValueError("unsupported feature kind")
        object.__setattr__(self, "parquet_path", Path(self.parquet_path))


def _manifest_state(view: MaterializedExecutionView | None = None, **kwargs):
    if view is not None:
        return {
            "version": view.version,
            "prepared_id": view.prepared_id,
            "packet_fingerprint": view.packet_fingerprint,
            "dataset_id": view.dataset_id,
            "symbol": view.symbol,
            "timeframe": view.timeframe,
            "row_count": view.row_count,
            "feature_ids": view.feature_ids,
            "feature_kinds": view.feature_kinds,
            "parquet_filename": view.parquet_path.name,
            "parquet_sha256": view.parquet_sha256,
        }
    return kwargs


def materialize_execution_view(
    dataset: DiscoveryDataset,
    packet: AIResearchPacket,
    *,
    directory: Path,
    prepared_id: str,
    dataset_id: str,
    symbol: str,
    timeframe: str,
) -> MaterializedExecutionView:
    """Materialize one compact, target-free execution table from a prepared dataset."""
    if type(dataset) is not DiscoveryDataset or type(packet) is not AIResearchPacket:
        raise TypeError("expected exact DiscoveryDataset and AIResearchPacket")
    packet.__post_init__()
    _validate_sha(prepared_id, "prepared_id")
    for value in (dataset_id, symbol, timeframe):
        _nonblank(value)

    feature_ids = tuple(packet.allowed_condition_feature_ids)
    kinds = dict(zip(packet.feature_ids, packet.feature_kinds))
    feature_kinds = tuple(kinds[field] for field in feature_ids)
    if len(set(dataset.predictor_field_ids)) != len(dataset.predictor_field_ids):
        raise ValueError("duplicate predictor ID")
    missing = [field for field in feature_ids if field not in dataset.predictor_field_ids]
    if missing:
        raise ValueError(f"packet feature missing from DiscoveryDataset: {missing[0]}")
    predictor_index = {field: dataset.predictor_field_ids.index(field) for field in feature_ids}

    row_indices = []
    timestamps = []
    reference_prices = []
    anchor_kinds = []
    geometry_family_ids = []
    geometry_level_ids = []
    feature_columns = {field: [] for field in feature_ids}

    for row_index, row in enumerate(dataset.rows):
        anchor = row.record.anchor
        row_indices.append(row_index)
        timestamps.append(_timestamp(anchor.evidence_end_utc))
        reference = _number(anchor.outcome_anchor.reference_price)
        if reference <= 0:
            raise ValueError("anchor reference price must be positive")
        reference_prices.append(reference)
        anchor_kinds.append(anchor.kind.value)
        geometry_family_ids.append(anchor.geometry_family_id)
        geometry_level_ids.append(anchor.level_id)
        for field in feature_ids:
            raw = row.predictors[predictor_index[field]].value
            feature_columns[field].append(_field_value(raw, kinds[field]))

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
        arrays.append(pa.array(
            feature_columns[field],
            type=pa.float64() if kind == "NUMERIC" else pa.string(),
        ))
        names.append(field)
    table = pa.Table.from_arrays(arrays, names=names)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / PARQUET_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    temporary = directory / f".{PARQUET_FILENAME}.{uuid4().hex}.tmp"
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        parquet_sha256 = sha256_file(temporary)
        state = _manifest_state(
            version=MATERIALIZED_EXECUTION_VIEW_VERSION,
            prepared_id=prepared_id,
            packet_fingerprint=packet.fingerprint,
            dataset_id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            row_count=len(dataset.rows),
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
            candidate = MaterializedExecutionView(
                version=MATERIALIZED_EXECUTION_VIEW_VERSION,
                prepared_id=prepared_id,
                packet_fingerprint=packet.fingerprint,
                dataset_id=dataset_id,
                symbol=symbol,
                timeframe=timeframe,
                row_count=len(dataset.rows),
                feature_ids=feature_ids,
                feature_kinds=feature_kinds,
                parquet_path=parquet_path,
                parquet_sha256=parquet_sha256,
                fingerprint=fingerprint,
            )
            if _manifest_state(existing) != _manifest_state(candidate):
                raise ValueError("existing execution view does not match requested lineage")
            return existing

        temporary.replace(parquet_path)
        manifest_path.write_text(canonical_json(document), encoding="utf-8")
        return load_execution_view(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_execution_view(directory: Path) -> MaterializedExecutionView:
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = document.pop("fingerprint", None)
    if fingerprint != sha256_canonical(document):
        raise ValueError("execution-view manifest fingerprint mismatch")
    if document.get("parquet_filename") != PARQUET_FILENAME:
        raise ValueError("unsupported execution-view parquet filename")
    parquet_path = directory / PARQUET_FILENAME
    if sha256_file(parquet_path) != document["parquet_sha256"]:
        raise ValueError("execution-view parquet checksum mismatch")
    return MaterializedExecutionView(
        version=document["version"],
        prepared_id=document["prepared_id"],
        packet_fingerprint=document["packet_fingerprint"],
        dataset_id=document["dataset_id"],
        symbol=document["symbol"],
        timeframe=document["timeframe"],
        row_count=document["row_count"],
        feature_ids=tuple(document["feature_ids"]),
        feature_kinds=tuple(document["feature_kinds"]),
        parquet_path=parquet_path,
        parquet_sha256=document["parquet_sha256"],
        fingerprint=fingerprint,
    )


def _sql_scalar(value, kind):
    return _field_value(value, kind)


def _sql_condition(condition, kinds, parameters):
    if type(condition) is AndCondition:
        return "(" + " AND ".join(_sql_condition(c, kinds, parameters) for c in condition.children) + ")"
    if type(condition) is not ComparisonCondition:
        raise ValueError("V1 supports only comparison and AND conditions")
    field = condition.reference.key
    kind = kinds[field]
    column = _quote(field)
    op = condition.operator.value

    if op in ("EQ", "NE"):
        if condition.value is None:
            return f"{column} IS {'NOT ' if op == 'NE' else ''}NULL"
        parameters.append(_sql_scalar(condition.value, kind))
        if op == "EQ":
            return f"{column} = ?"
        # Python V1 semantics: None != non-None is True.
        return f"({column} IS NULL OR {column} <> ?)"

    if op in ("GT", "GTE", "LT", "LTE"):
        parameters.append(_sql_scalar(condition.value, kind))
        token = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}[op]
        return f"({column} IS NOT NULL AND {column} {token} ?)"

    values = tuple(condition.value)
    nonnull = [value for value in values if value is not None]
    contains_null = len(nonnull) != len(values)
    encoded = [_sql_scalar(value, kind) for value in nonnull]

    if op == "IN":
        clauses = []
        if contains_null:
            clauses.append(f"{column} IS NULL")
        for value in encoded:
            parameters.append(value)
            clauses.append(f"{column} = ?")
        return "(" + " OR ".join(clauses) + ")"

    if op == "NOT_IN":
        if not nonnull:
            return f"{column} IS NOT NULL"
        comparisons = []
        for value in encoded:
            parameters.append(value)
            comparisons.append(f"{column} <> ?")
        nonnull_clause = "(" + " AND ".join(comparisons) + ")"
        if contains_null:
            return f"({column} IS NOT NULL AND {nonnull_clause})"
        # Python V1 semantics: None is not in a list that contains no None.
        return f"({column} IS NULL OR {nonnull_clause})"

    raise ValueError("unsupported comparison operator")


def _query_rule(connection, view, rule, kinds):
    parameters = []
    where = _sql_condition(rule.condition, kinds, parameters)
    path = str(view.parquet_path.resolve()).replace("'", "''")
    selected = ", ".join(_quote(name) for name in _BASE_COLUMNS)
    sql = (
        f"SELECT {selected} FROM read_parquet('{path}') "
        f"WHERE {where} ORDER BY {_quote('source_row_index')}"
    )
    return connection.execute(sql, parameters).fetchall()


def build_materialized_strategy_intents(
    view: MaterializedExecutionView,
    compiled: CompiledHypothesis,
    packet: AIResearchPacket,
    *,
    pip_size: float,
) -> CompiledStrategyIntentBatch:
    """Execute the frozen V1 hypothesis directly against the compact Parquet view."""
    if type(view) is not MaterializedExecutionView or type(compiled) is not CompiledHypothesis or type(packet) is not AIResearchPacket:
        raise TypeError("expected exact materialized view, compiled hypothesis, and packet types")
    packet.__post_init__()
    if view.packet_fingerprint != packet.fingerprint:
        raise ValueError("execution view must match research packet")
    if tuple(view.feature_ids) != tuple(packet.allowed_condition_feature_ids):
        raise ValueError("execution-view feature schema does not match packet")
    if tuple(view.feature_kinds) != tuple(packet.feature_kinds):
        raise ValueError("execution-view feature kinds do not match packet")
    if compiled.research_packet_fingerprint != packet.fingerprint or compiled.target_field_id != packet.target_field_id:
        raise ValueError("compiled hypothesis must match packet")
    _nonblank(view.dataset_id)
    _nonblank(view.symbol)
    pip_size = _number(pip_size)
    if pip_size <= 0:
        raise ValueError("pip size must be positive")

    kinds = dict(zip(packet.feature_ids, packet.feature_kinds))
    referenced = set()
    for rule in compiled.strategy_spec.rules:
        _check_condition(rule.condition, kinds, view.feature_ids, referenced)
        if type(rule.entry) is not MarketEntry or type(rule.stop) is not FixedPipsStop or type(rule.target) is not RiskMultipleTarget:
            raise ValueError("V1 requires market entry, fixed-pips stop, and risk-multiple target")

    matches = []
    with duckdb.connect(database=":memory:") as connection:
        for rule_order, rule in enumerate(compiled.strategy_spec.rules):
            for source in _query_rule(connection, view, rule, kinds):
                matches.append((source[0], rule_order, source, rule))
    matches.sort(key=lambda item: (item[0], item[1]))

    intents = []
    matched_indices = []
    for row_index, _, source, rule in matches:
        _, timestamp, reference, anchor_kind, geometry_family_id, geometry_level_id = source
        timestamp = _timestamp(timestamp)
        reference = _number(reference)
        if reference <= 0:
            raise ValueError("anchor reference price must be positive")
        risk = rule.stop.pips * pip_size
        direction = Direction(rule.entry.side.value)
        sign = 1 if direction == Direction.LONG else -1
        stop = reference - sign * risk
        target = reference + sign * risk * rule.target.multiple
        _geometry(reference, stop, target, direction)
        metadata = dict(
            execution_adapter="COMPILED_DISCOVERY_DSL_V1",
            hypothesis_id=compiled.hypothesis_id,
            compiled_hypothesis_fingerprint=compiled.fingerprint,
            strategy_spec_fingerprint=compiled.strategy_spec_fingerprint,
            research_packet_fingerprint=packet.fingerprint,
            rule_id=rule.rule_id,
            source_row_index=row_index,
            anchor_kind=anchor_kind,
            geometry_family_id=geometry_family_id,
            geometry_level_id=geometry_level_id,
        )
        intents.append(TradeIntent(
            trade_intent_id=_intent_id(compiled.fingerprint, view.dataset_id, row_index, timestamp, rule.rule_id),
            strategy_id=compiled.strategy_spec.strategy_id,
            experiment_id=compiled.hypothesis_id,
            dataset_id=view.dataset_id,
            symbol=view.symbol,
            direction=direction,
            signal_timestamp=timestamp,
            requested_entry_type=EntryType.MARKET_CLOSE,
            requested_entry_price=reference,
            stop_loss=stop,
            take_profit=target,
            metadata=metadata,
        ))
        if not matched_indices or matched_indices[-1] != row_index:
            matched_indices.append(row_index)

    state = dict(
        version=COMPILED_STRATEGY_EXECUTION_VERSION,
        compiled_hypothesis_fingerprint=compiled.fingerprint,
        strategy_spec_fingerprint=compiled.strategy_spec_fingerprint,
        research_packet_fingerprint=packet.fingerprint,
        hypothesis_id=compiled.hypothesis_id,
        strategy_id=compiled.strategy_spec.strategy_id,
        dataset_id=view.dataset_id,
        symbol=view.symbol,
        pip_size=pip_size,
        source_row_count=view.row_count,
        matched_row_count=len(matched_indices),
        intent_count=len(intents),
        matched_source_row_indices=tuple(matched_indices),
        intents=tuple(intents),
    )
    payload = {**state, "intents": [intent.model_dump(mode="json") for intent in intents]}
    return CompiledStrategyIntentBatch(**state, fingerprint=_digest(payload))
