from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction, EntryType, ExecutionConfig
from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
from sandbox.market_state.discovery_projection import DiscoveryRow, DiscoveryValue
from sandbox.partition.models import PartitionRole
from sandbox.research import strategy_runner as runner
from sandbox.strategies.dsl.conditions import AndCondition, ComparisonCondition
from test_discovery_dataset import TIME, record


def dataset(category="TOUCH_TRANSITION", number=1.0, target=999.0):
    row = DiscoveryRow(
        record(), (),
        (DiscoveryValue("x.anchor_kind", category),
         DiscoveryValue("x.geometry.coordinate", number)),
        (DiscoveryValue("y.h4.close_return_fraction", target),),
    )
    return assemble_discovery_dataset((row,))


def candles(*, direction="LONG", entry=100.0):
    if direction == "LONG":
        high, low, close = entry + 2.1, entry - .5, entry + 2.0
    else:
        high, low, close = entry + .5, entry - 2.1, entry - 2.0
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime([TIME, TIME + timedelta(minutes=15)], utc=True),
        "open": [entry, close], "high": [high, close + .1],
        "low": [low, close - .1], "close": [close, close],
        "spread": [0, 0],
    })


def strategy(*, direction="LONG", operator="=", value="TOUCH_TRANSITION",
             field_id="x.anchor_kind"):
    return runner.StrategyRunnerSpec.model_validate({
        "strategy_id": "TEST_STRATEGY_V1", "partition_id": "PART-DISCOVERY",
        "direction": direction,
        "conditions": [{"field_id": field_id, "operator": operator, "value": value}],
        "entry": {"type": "NEXT_OPEN"},
        "stop_loss": {"type": "FIXED_PIPS", "value": 10.0},
        "take_profit": {"type": "R_MULTIPLE", "value": 2.0},
    })


def execute(tmp_path, spec=None, data=None, market=None):
    spec = spec or strategy()
    return runner.run_strategy(
        spec, data or dataset(), market if market is not None else candles(direction=spec.direction.value),
        dataset_id=spec.partition_id, symbol="EURUSD", pip_size=.1,
        execution_config=ExecutionConfig(candle_interval_seconds=900), root=tmp_path,
    )


def test_valid_strategy_loads_and_builds_existing_dsl(tmp_path):
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(strategy().model_dump(mode="json")), encoding="utf-8")
    loaded = runner.load_strategy(path, available_feature_ids=dataset().predictor_field_ids)
    expression = loaded.condition_expression()
    assert loaded.strategy_id == "TEST_STRATEGY_V1"
    assert isinstance(expression, AndCondition)
    assert isinstance(expression.children[0], ComparisonCondition)
    assert expression.children[0].reference.namespace == "feature"


@pytest.mark.parametrize("location", ["top", "condition", "entry", "stop", "target"])
def test_unknown_json_keys_rejected(tmp_path, location):
    payload = strategy().model_dump(mode="json")
    target = payload if location == "top" else payload["conditions"][0] if location == "condition" else payload[location if location == "entry" else "stop_loss" if location == "stop" else "take_profit"]
    target["unknown"] = True
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs"):
        runner.load_strategy(path)


def test_invalid_feature_id_rejected_against_actual_schema():
    with pytest.raises(ValueError, match="does not exist"):
        runner.build_strategy_intents(
            strategy(field_id="x.not_in_dataset"), dataset(), candles(),
            dataset_id="PART-DISCOVERY", symbol="EURUSD", pip_size=.1)


@pytest.mark.parametrize("direction, expected_stop, expected_target", [
    ("LONG", 99.0, 102.0), ("SHORT", 101.0, 98.0),
])
def test_long_and_short_signal_geometry(direction, expected_stop, expected_target):
    spec = strategy(direction=direction)
    intents = runner.build_strategy_intents(
        spec, dataset(), candles(direction=direction), dataset_id=spec.partition_id,
        symbol="EURUSD", pip_size=.1)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.direction == Direction(direction)
    assert intent.requested_entry_type == EntryType.MARKET_NEXT_OPEN
    assert intent.stop_loss == pytest.approx(expected_stop)
    assert intent.take_profit == pytest.approx(expected_target)


@pytest.mark.parametrize("operator, threshold, expected", [
    ("<", 1.1, True), ("<=", 1.0, True), (">", .9, True), (">=", 1.0, True),
    ("<", 1.0, False), (">", 1.0, False),
])
def test_numeric_comparisons(operator, threshold, expected):
    spec = strategy(field_id="x.geometry.coordinate", operator=operator, value=threshold)
    intents = runner.build_strategy_intents(
        spec, dataset(number=1.0), candles(), dataset_id=spec.partition_id,
        symbol="EURUSD", pip_size=.1)
    assert bool(intents) is expected


def test_categorical_comparison_and_failed_conditions():
    assert runner.build_strategy_intents(
        strategy(), dataset(), candles(), dataset_id="D", symbol="EURUSD", pip_size=.1)
    assert runner.build_strategy_intents(
        strategy(), dataset(category="UNTOUCHED_CHECKPOINT"), candles(),
        dataset_id="D", symbol="EURUSD", pip_size=.1) == ()


def test_causal_predictors_only_and_no_future_target_access(monkeypatch):
    source = dataset(target=-999.0)
    original = DiscoveryRow.__getattribute__

    def guarded(self, name):
        if name == "targets":
            raise AssertionError("future target accessed")
        return original(self, name)

    monkeypatch.setattr(DiscoveryRow, "__getattribute__", guarded)
    intents = runner.build_strategy_intents(
        strategy(), source, candles(), dataset_id="D", symbol="EURUSD", pip_size=.1)
    assert len(intents) == 1 and intents[0].signal_timestamp == TIME


def test_next_open_entry_semantics(tmp_path):
    result = execute(tmp_path, market=candles(entry=123.0))
    trade = result.engine_result.ledger[0]
    assert result.intents[0].signal_timestamp == TIME
    assert trade.entry_timestamp == TIME
    assert trade.entry_price == pytest.approx(123.0)


def test_deterministic_repeated_runs_and_identity(tmp_path):
    first = execute(tmp_path)
    before = {path.name: path.read_bytes() for path in first.result_directory.iterdir()}
    second = execute(tmp_path)
    assert first.run_id == second.run_id
    assert first.run_fingerprint == second.run_fingerprint
    assert first.intents == second.intents
    assert first.engine_result == second.engine_result
    assert before == {path.name: path.read_bytes() for path in second.result_directory.iterdir()}


def test_trade_ledger_outputs_and_metrics(tmp_path):
    result = execute(tmp_path)
    required = {"strategy.json", "manifest.json", "trades.csv", "metrics.json", "report.md"}
    assert {path.name for path in result.result_directory.iterdir()} == required
    trades = pd.read_csv(result.result_directory / "trades.csv")
    assert tuple(trades.columns) == runner._TRADE_COLUMNS
    assert len(trades) == 1 and trades.iloc[0].exit_reason == "TAKE_PROFIT"
    metrics = json.loads((result.result_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["trade_count"] == metrics["wins"] == metrics["long_count"] == 1
    assert metrics["losses"] == metrics["short_count"] == 0
    assert metrics["gross_r"] == pytest.approx(2.0)
    assert metrics["net_r"] == pytest.approx(2.0)
    assert metrics["expectancy_r"] == pytest.approx(2.0)
    assert metrics["yearly_results"][0]["year"] == TIME.year


def test_zero_trade_strategy_is_safe(tmp_path):
    result = execute(tmp_path, data=dataset(category="UNTOUCHED_CHECKPOINT"))
    assert result.intents == () and result.engine_result.ledger == ()
    assert result.metrics["trade_count"] == result.metrics["wins"] == 0
    assert result.metrics["profit_factor"] == 0.0
    assert len(pd.read_csv(result.result_directory / "trades.csv")) == 0


def test_existing_backtest_engine_is_invoked(tmp_path, monkeypatch):
    calls = []

    class SpyEngine(BacktestEngine):
        def run(self, intents, market):
            calls.append((tuple(intents), market))
            return super().run(calls[-1][0], market)

    monkeypatch.setattr(runner, "BacktestEngine", SpyEngine)
    result = execute(tmp_path)
    assert len(calls) == 1 and calls[0][0] == result.intents


def test_models_frozen_and_unsupported_execution_rejected():
    spec = strategy()
    with pytest.raises(ValidationError):
        runner.StrategyRunnerSpec.model_validate({**spec.model_dump(mode="json"), "entry": {"type": "CLOSE"}})
    with pytest.raises(ValidationError):
        runner.StrategyRunnerSpec.model_validate({**spec.model_dump(mode="json"), "conditions": []})
    with pytest.raises(ValidationError):
        spec.strategy_id = "OTHER"


def test_example_uses_confirmed_discovery_features():
    available = dataset().predictor_field_ids
    loaded = runner.load_strategy(Path("experiments/example_strategy_v1.json"),
                                  available_feature_ids=available)
    assert {condition.field_id for condition in loaded.conditions} <= set(available)


def test_protected_partition_rejected_before_build(monkeypatch):
    class Service:
        def get_partition(self, partition_id):
            return type("Manifest", (), {"role": PartitionRole.VALIDATION, "timeframe": "M15"})()

    monkeypatch.setattr(runner, "build_discovery_record_population",
                        lambda *_args, **_kwargs: pytest.fail("protected partition was read"))
    with pytest.raises(ValueError, match="Validation partition rejected"):
        runner.run_partition_strategy(Service(), strategy())


def test_cli_summary_contains_required_metrics(tmp_path, capsys):
    result = execute(tmp_path)
    runner._print_summary(result)
    output = capsys.readouterr().out
    assert "Strategy: TEST_STRATEGY_V1" in output
    assert "Trades: 1" in output and "Net R: 2.00" in output


def partition(partition_id, source_dataset_id, role=PartitionRole.DISCOVERY):
    return SimpleNamespace(partition_id=partition_id, source_dataset_id=source_dataset_id,
                           role=role, timeframe="M15")


class PartitionLookup:
    def __init__(self, *manifests):
        self.manifests = {item.partition_id: item for item in manifests}

    def get_partition(self, partition_id):
        return self.manifests.get(partition_id)


def without_partition():
    return runner.StrategyRunnerSpec.model_validate({
        key: value for key, value in strategy().model_dump(mode="json").items()
        if key != "partition_id"
    })


def test_dataset_resolves_one_discovery_partition(monkeypatch):
    found = partition("PART-ONE", "DATASET-1")
    monkeypatch.setattr(runner, "_partitions_for_dataset", lambda _service, _dataset: (found,))
    resolved = runner.resolve_strategy_partition(PartitionLookup(found), without_partition(), "DATASET-1")
    assert resolved.partition_id == "PART-ONE"


def test_dataset_with_no_discovery_partition_fails_clearly(monkeypatch):
    monkeypatch.setattr(runner, "_partitions_for_dataset", lambda _service, _dataset: ())
    with pytest.raises(ValueError, match="no Discovery partition exists.*create a Discovery partition first"):
        runner.resolve_strategy_partition(PartitionLookup(), without_partition(), "DATASET-1")


def test_dataset_with_multiple_discovery_partitions_is_ambiguous(monkeypatch):
    found = (partition("PART-A", "DATASET-1"), partition("PART-B", "DATASET-1"))
    monkeypatch.setattr(runner, "_partitions_for_dataset", lambda _service, _dataset: found)
    with pytest.raises(ValueError, match="multiple Discovery partitions.*specify partition_id explicitly"):
        runner.resolve_strategy_partition(PartitionLookup(*found), without_partition(), "DATASET-1")


@pytest.mark.parametrize("role, message", [
    (PartitionRole.VALIDATION, "Validation partition rejected"),
    (PartitionRole.FINAL_TEST, "Final Holdout partition rejected"),
])
def test_dataset_with_only_protected_partition_is_rejected(monkeypatch, role, message):
    found = partition("PART-PROTECTED", "DATASET-1", role)
    monkeypatch.setattr(runner, "_partitions_for_dataset", lambda _service, _dataset: (found,))
    with pytest.raises(ValueError, match=message):
        runner.resolve_strategy_partition(PartitionLookup(found), without_partition(), "DATASET-1")


@pytest.mark.parametrize("role, message", [
    (PartitionRole.VALIDATION, "Validation partition rejected"),
    (PartitionRole.FINAL_TEST, "Final Holdout partition rejected"),
])
def test_explicit_protected_partition_is_rejected(role, message):
    found = partition("PART-DISCOVERY", "DATASET-1", role)
    with pytest.raises(ValueError, match=message):
        runner.resolve_strategy_partition(PartitionLookup(found), strategy())


def test_conflicting_json_partition_and_dataset_id_rejected():
    found = partition("PART-DISCOVERY", "DATASET-OTHER")
    with pytest.raises(ValueError, match="conflicts with --dataset-id"):
        runner.resolve_strategy_partition(PartitionLookup(found), strategy(), "DATASET-1")


def test_existing_json_partition_workflow_is_unchanged():
    spec = strategy()
    found = partition(spec.partition_id, "DATASET-1")
    assert runner.resolve_strategy_partition(PartitionLookup(found), spec) is spec


def test_cli_dataset_id_reports_missing_discovery_partition(tmp_path, monkeypatch, capsys):
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(without_partition().model_dump(mode="json", exclude_none=True)),
                    encoding="utf-8")
    monkeypatch.setattr(runner.Settings, "load",
                        lambda: SimpleNamespace(catalog_path=tmp_path / "catalog.sqlite"))
    monkeypatch.setattr(runner, "ResearchRegistry", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "PartitionService", lambda _registry: PartitionLookup())
    monkeypatch.setattr(runner, "_partitions_for_dataset", lambda _service, _dataset: ())
    assert runner.main([str(path), "--dataset-id", "DATASET-1"]) == 2
    assert "no Discovery partition exists" in capsys.readouterr().err


def test_run_identity_binds_market_data_even_when_no_trades(tmp_path):
    source = dataset(category="UNTOUCHED_CHECKPOINT")
    first = execute(tmp_path, data=source)
    changed = candles()
    changed.loc[1, "high"] += 1
    second = execute(tmp_path, data=source, market=changed)
    assert first.intents == second.intents == ()
    assert first.run_fingerprint != second.run_fingerprint


def test_real_m1_partition_pipeline_uses_causal_builder_and_access_ledger(tmp_path):
    from test_discovery_record_builder import market_frame, partition as make_partition
    m15 = market_frame(80, touches=True)
    m1 = pd.concat([
        m15.assign(timestamp_utc=m15.timestamp_utc + timedelta(minutes=minute))
        for minute in range(15)
    ]).sort_values("timestamp_utc").reset_index(drop=True)
    service, manifest = make_partition(tmp_path, m1, timeframe="M1")
    spec = strategy().model_copy(update={"partition_id": manifest.partition_id})
    result = runner.run_partition_strategy(service, spec, root=tmp_path/"runs", pip_size=.1)
    assert result.intents
    assert result.engine_result.execution_config.candle_interval_seconds == 60
    assert service.verify_access_ledger()
    with service.registry.catalog.connection() as con:
        rows = con.execute("SELECT actor,access_context FROM partition_access_ledger").fetchall()
    assert ("discovery-record-builder", "DISCOVERY_CONTEXT") in [tuple(row) for row in rows]
    assert ("strategy-runner-v1", "DISCOVERY_CONTEXT") in [tuple(row) for row in rows]


def test_cli_reports_research_access_denial(tmp_path, monkeypatch, capsys):
    from sandbox.research.errors import ResearchError
    path = tmp_path/"strategy.json"
    path.write_text(strategy().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(runner.Settings, "load", lambda: SimpleNamespace(catalog_path=tmp_path/"catalog.sqlite"))
    monkeypatch.setattr(runner, "ResearchRegistry", lambda *a, **kw: object())
    monkeypatch.setattr(runner, "PartitionService", lambda _: object())
    def denied(*args, **kwargs):
        raise ResearchError("ACCESS_DENIED")
    monkeypatch.setattr(runner, "run_partition_strategy", denied)
    assert runner.main([str(path)]) == 2
    assert "ACCESS_DENIED" in capsys.readouterr().err
