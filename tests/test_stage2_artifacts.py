import json
from pathlib import Path

import pandas as pd

from sandbox.catalog import Catalog
from sandbox.execution.artifacts import store_run
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, TradeIntent
from sandbox.provenance import sha256_file


def test_run_artifacts_catalog_migration_and_deterministic_rewrite():
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2025-01-01 00:00Z", "2025-01-01 00:01Z"], utc=True), "open": [99, 100], "high": [101, 121], "low": [98, 99], "close": [100, 120], "tick_volume": [1, 1], "spread": [0, 0], "real_volume": [0, 0]})
    item = TradeIntent(trade_intent_id="artifact", strategy_id="s", experiment_id="e", dataset_id="d", symbol="X", direction=Direction.LONG, signal_timestamp=pd.Timestamp("2025-01-01 00:00:30Z").to_pydatetime(), requested_entry_type=EntryType.MARKET_NEXT_OPEN, stop_loss=90, take_profit=120)
    result = BacktestEngine(ExecutionConfig()).run([item], frame)
    root = Path("tests/runtime/stage2-results")
    first = store_run(result, root); second = store_run(result, root)
    assert first == second and first.ledger_checksum == sha256_file(first.ledger_path)
    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["net_R"] == 2.0
    catalog = Catalog(Path("tests/runtime/stage2-catalog.db")); catalog.initialize()
    catalog.add_research_run(first.run_id, first.run_fingerprint, "d", result.execution_engine_version, result.ledger_schema_version, result.metrics_version, result.execution_config.model_dump(mode="json"), str(first.ledger_path), str(first.summary_path))
    assert catalog.research_run(first.run_id)["run_fingerprint"] == result.run_fingerprint


def test_missing_market_interval_is_explicitly_counted():
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2025-01-03 21:59Z", "2025-01-03 22:00Z", "2025-01-05 22:00Z"], utc=True), "open": [99, 100, 101], "high": [101, 105, 121], "low": [98, 95, 99], "close": [100, 102, 120], "tick_volume": [1,1,1], "spread": [0,0,0], "real_volume": [0,0,0]})
    item = TradeIntent(trade_intent_id="gap", strategy_id="s", experiment_id="e", dataset_id="d", symbol="X", direction=Direction.LONG, signal_timestamp=pd.Timestamp("2025-01-03 21:59:30Z").to_pydatetime(), requested_entry_type=EntryType.MARKET_NEXT_OPEN, stop_loss=90, take_profit=120)
    record = BacktestEngine(ExecutionConfig()).run([item], frame).ledger[0]
    assert record.market_gap_count == 1
