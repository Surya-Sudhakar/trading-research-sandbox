import pandas as pd

from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, PositionStatus, TradeIntent


def market():
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 00:01Z", "2024-01-01 00:02Z", "2024-01-01 00:03Z"], utc=True),
        "open": [100.0, 100.0, 99.0, 101.0],
        "high": [101.0, 101.0, 102.0, 103.0],
        "low": [100.0, 99.5, 98.0, 100.0],
        "close": [100.0, 100.0, 101.0, 102.0],
        "spread": [0.0, 0.0, 0.0, 0.0],
    })


def test_buy_limit_fills_when_price_reaches_limit_before_expiry():
    intent = TradeIntent(
        trade_intent_id="limit-fill",
        strategy_id="TEST",
        experiment_id="TEST",
        dataset_id="DATA",
        symbol="EURUSD",
        direction=Direction.LONG,
        signal_timestamp=pd.Timestamp("2024-01-01 00:01Z").to_pydatetime(),
        requested_entry_type=EntryType.LIMIT,
        requested_entry_price=99.0,
        expires_at=pd.Timestamp("2024-01-01 00:03Z").to_pydatetime(),
        stop_loss=98.0,
        take_profit=102.0,
    )
    row = BacktestEngine(ExecutionConfig(candle_interval_seconds=60)).run([intent], market()).ledger[0]
    assert row.rejection_reason is None
    assert row.entry_price == 99.0
    assert row.entry_timestamp == pd.Timestamp("2024-01-01 00:02Z").to_pydatetime()
    assert row.status == PositionStatus.CLOSED


def test_buy_limit_does_not_fill_after_expiry():
    intent = TradeIntent(
        trade_intent_id="limit-expiry",
        strategy_id="TEST",
        experiment_id="TEST",
        dataset_id="DATA",
        symbol="EURUSD",
        direction=Direction.LONG,
        signal_timestamp=pd.Timestamp("2024-01-01 00:00Z").to_pydatetime(),
        requested_entry_type=EntryType.LIMIT,
        requested_entry_price=99.0,
        expires_at=pd.Timestamp("2024-01-01 00:01Z").to_pydatetime(),
        stop_loss=98.0,
        take_profit=102.0,
    )
    row = BacktestEngine(ExecutionConfig(candle_interval_seconds=60)).run([intent], market()).ledger[0]
    assert row.entry_timestamp is None
    assert row.entry_price is None
