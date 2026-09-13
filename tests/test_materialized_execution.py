from pathlib import Path

import pyarrow.parquet as pq
import pytest

from test_compiled_strategy_execution import artifacts, dataset, execute
from sandbox.research.materialized_execution import (
    MATERIALIZED_EXECUTION_VIEW_VERSION,
    build_materialized_strategy_intents,
    load_execution_view,
    materialize_execution_view,
)


PREPARED_ID = "a" * 64


def view(tmp_path, source=None, parts=None):
    compiled, packet = artifacts() if parts is None else parts
    source = dataset() if source is None else source
    result = materialize_execution_view(
        source,
        packet,
        directory=tmp_path / "execution_view",
        prepared_id=PREPARED_ID,
        dataset_id="DISCOVERY_EVENTS",
        symbol="EURUSD",
        timeframe="M15",
    )
    return result, compiled, packet, source


def run_materialized(tmp_path, source=None, parts=None, pip_size=.0001):
    v, compiled, packet, _ = view(tmp_path, source, parts)
    return build_materialized_strategy_intents(v, compiled, packet, pip_size=pip_size)


def test_materialized_view_is_compact_target_free_and_reloadable(tmp_path):
    v, _, packet, source = view(tmp_path, dataset((("BULLISH", 20.), ("BEARISH", 30.))))
    assert v.version == MATERIALIZED_EXECUTION_VIEW_VERSION
    assert v.row_count == len(source.rows) == 2
    assert v.feature_ids == packet.allowed_condition_feature_ids
    assert v.feature_kinds == packet.feature_kinds
    schema = pq.read_schema(v.parquet_path)
    assert "source_row_index" in schema.names
    assert "reference_price" in schema.names
    assert all(field in schema.names for field in packet.allowed_condition_feature_ids)
    assert not any(name.startswith("y.") for name in schema.names)
    assert load_execution_view(v.parquet_path.parent) == v


@pytest.mark.parametrize("operator,value", [
    ("LT", 30.), ("LTE", 30.), ("GT", 30.), ("GTE", 30.),
    ("EQ", 20.), ("NE", 20.), ("EQ", None), ("NE", None),
    ("IN", [20., None]), ("NOT_IN", [20., None]),
])
def test_numeric_operator_parity(tmp_path, operator, value):
    source = dataset(tuple(("BULLISH", v) for v in (20, 30., 40., None)))
    parts = artifacts(operator, value)
    assert run_materialized(tmp_path, source, parts) == execute(source, parts)


@pytest.mark.parametrize("operator,value", [
    ("EQ", False), ("NE", False), ("IN", [False, "False"]),
    ("NOT_IN", [False, "False"]), ("EQ", None),
])
def test_typed_categorical_parity(tmp_path, operator, value):
    source = dataset(tuple((v, 20.) for v in (False, 0, 0., "False", None)))
    parts = artifacts(operator, value, field=0)
    assert run_materialized(tmp_path, source, parts) == execute(source, parts)


def test_multiple_rule_short_parity(tmp_path):
    source = dataset()
    source = source.__class__(
        rows=(source.rows[0], source.rows[0]),
        identity_field_ids=source.identity_field_ids,
        predictor_field_ids=source.predictor_field_ids,
        target_field_ids=source.target_field_ids,
    )
    parts = artifacts(side="SHORT", two_rules=True)
    assert run_materialized(tmp_path, source, parts) == execute(source, parts)


def test_manifest_and_parquet_tampering_are_detected(tmp_path):
    v, _, _, _ = view(tmp_path)
    manifest = v.parquet_path.parent / "execution_view.json"
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(original.replace(PREPARED_ID, "b" * 64), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint"):
        load_execution_view(v.parquet_path.parent)
    manifest.write_text(original, encoding="utf-8")
    with v.parquet_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_execution_view(v.parquet_path.parent)
