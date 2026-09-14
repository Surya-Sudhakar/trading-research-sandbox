"""Regression tests for large-batch BacktestEngine setup costs.

These tests protect the deterministic semantics while preventing accidental
reintroduction of per-intent market canonicalization or quadratic duplicate
scanning.
"""
from datetime import timezone

import pandas as pd

from sandbox.execution import engine as execution_engine
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, PositionStatus, TradeIntent


UTC = timezone.utc


def _market():
    return pd.DataFrame([
        ("2024-01-01 00:00Z", 100.0, 101.0, 99.0, 100.0, 0.0),
        ("2024-01-01 00:01Z", 100.0, 102.0, 98.0, 101.0, 0.0),
        ("2024-01-01 00:02Z", 101.0, 103.0, 100.0, 102.0, 0.0),
    ], columns=["timestamp_utc", "open", "high", "low", "close", "spread"])


def _intent(ident, signal="2024-01-01 00:01Z"):
    return TradeIntent(
        trade_intent_id=ident,
        strategy_id="SCALING_TEST",
        experiment_id="SCALING_TEST",
        dataset_id="DATASET",
        symbol="TEST",
        direction=Direction.LONG,
        signal_timestamp=pd.Timestamp(signal).to_pydatetime(),
        requested_entry_type=EntryType.MARKET_CLOSE,
        stop_loss=99.0,
        take_profit=103.0,
    )


def test_single_market_is_canonicalized_once_for_many_intents(monkeypatch):
    calls = 0
    original = execution_engine.canonical_market_data

    def counted(frame):
        nonlocal calls
        calls += 1
        return original(frame)

    monkeypatch.setattr(execution_engine, "canonical_market_data", counted)
    intents = [_intent(f"i-{i}") for i in range(100)]
    execution_engine.BacktestEngine(ExecutionConfig(candle_interval_seconds=60)).run(intents, _market())
    assert calls == 1


def test_duplicate_trade_intent_semantics_preserved():
    duplicated = _intent("dup")
    unique = _intent("unique")
    result = execution_engine.BacktestEngine(ExecutionConfig(candle_interval_seconds=60)).run(
        [duplicated, duplicated, unique], _market()
    )
    duplicate_records = [row for row in result.ledger if row.trade_intent_id == "dup"]
    assert len(duplicate_records) == 2
    assert all(row.status == PositionStatus.REJECTED for row in duplicate_records)
    assert all(row.rejection_reason == "DUPLICATE_TRADE_INTENT_ID" for row in duplicate_records)
    assert next(row for row in result.ledger if row.trade_intent_id == "unique").rejection_reason is None


def test_microsecond_market_resolution_preserves_execution_semantics():
    config = ExecutionConfig(candle_interval_seconds=60)
    baseline = execution_engine.BacktestEngine(config).run(
        [_intent("baseline-resolution")], _market()
    ).ledger[0]

    market = _market()
    market["timestamp_utc"] = pd.Series(
        pd.DatetimeIndex(pd.to_datetime(market["timestamp_utc"], utc=True)).as_unit("us")
    )
    microsecond = execution_engine.BacktestEngine(config).run(
        [_intent("microsecond-resolution")], market
    ).ledger[0]

    assert microsecond.rejection_reason is None
    assert microsecond.status == baseline.status
    assert microsecond.exit_reason == baseline.exit_reason
    assert microsecond.entry_timestamp == baseline.entry_timestamp
    assert microsecond.entry_price == baseline.entry_price
    assert microsecond.exit_timestamp == baseline.exit_timestamp
    assert microsecond.exit_price == baseline.exit_price
    assert microsecond.gross_r == baseline.gross_r
    assert microsecond.net_r == baseline.net_r
