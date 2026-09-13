import pandas as pd
import pytest

from sandbox.execution.engine import BacktestEngine
from sandbox.execution.metrics import summarize, symbol_breakdown, yearly_breakdown
from sandbox.execution.models import CommissionModel, Direction, EntryType, ExecutionConfig, PositionStatus, SlippageModel, TradeIntent


def candles():
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 00:01Z", "2024-01-01 00:02Z"], utc=True),
        "open": [99, 100, 101], "high": [101, 105, 121], "low": [98, 95, 99], "close": [100, 102, 120],
        "tick_volume": [1, 1, 1], "spread": [0, 0, 0], "real_volume": [0, 0, 0],
    })


def intent(ident="id", entry=EntryType.MARKET_NEXT_OPEN, signal="2024-01-01 00:00:30Z"):
    return TradeIntent(trade_intent_id=ident, strategy_id="s", experiment_id="e", dataset_id="d", symbol="X", direction=Direction.LONG, signal_timestamp=pd.Timestamp(signal).to_pydatetime(), requested_entry_type=entry, stop_loss=90, take_profit=120)


def test_deterministic_rerun_and_input_sort_invariance():
    config = ExecutionConfig()
    first = BacktestEngine(config).run([intent()], candles())
    second = BacktestEngine(config).run([intent()], candles().iloc[::-1])
    assert first.run_fingerprint == second.run_fingerprint
    pd.testing.assert_frame_equal(first.frame(), second.frame())


def test_duplicate_ids_are_all_rejected():
    result = BacktestEngine(ExecutionConfig()).run([intent(), intent()], candles())
    assert all(x.rejection_reason == "DUPLICATE_TRADE_INTENT_ID" for x in result.ledger)


def test_unsupported_order_types_never_approximated():
    for entry in (EntryType.LIMIT, EntryType.STOP):
        assert BacktestEngine(ExecutionConfig()).run([intent(entry=entry)], candles()).ledger[0].rejection_reason == "UNSUPPORTED_ENTRY_TYPE"


def test_lookahead_and_impossible_market_close_are_rejected():
    outside = BacktestEngine(ExecutionConfig()).run([intent(signal="2023-12-31 23:59Z")], candles()).ledger[0]
    impossible = BacktestEngine(ExecutionConfig()).run([intent(entry=EntryType.MARKET_CLOSE, signal="2024-01-01 00:00:30Z")], candles()).ledger[0]
    assert outside.rejection_reason == "SIGNAL_OUTSIDE_DATASET"
    assert impossible.rejection_reason == "IMPOSSIBLE_SIGNAL_TIMESTAMP"


def test_increasing_commission_and_slippage_cannot_improve_net_pnl():
    base = BacktestEngine(ExecutionConfig()).run([intent()], candles()).ledger[0]
    commission = BacktestEngine(ExecutionConfig(commission_model=CommissionModel.FIXED_PER_TRADE, fixed_commission=2)).run([intent()], candles()).ledger[0]
    slip = BacktestEngine(ExecutionConfig(slippage_model=SlippageModel.FIXED_PRICE_SLIPPAGE, fixed_slippage_price=1)).run([intent()], candles()).ledger[0]
    assert commission.net_pnl <= base.net_pnl and slip.net_pnl <= base.net_pnl


def test_metrics_and_breakdowns_derive_from_ledger():
    winning = intent("win")
    invalid = TradeIntent(**{**intent("bad").model_dump(), "stop_loss": 101})
    ledger = BacktestEngine(ExecutionConfig()).run([winning, invalid], candles()).frame()
    metrics = summarize(ledger)
    assert metrics["total_trade_intents"] == 2 and metrics["resolved_trades"] == 1
    assert metrics["wins"] == 1 and metrics["net_R"] == pytest.approx(2)
    assert yearly_breakdown(ledger)[0]["year"] == 2024
    assert symbol_breakdown(ledger)[0]["symbol"] == "X"


def test_audit_hooks_receive_lifecycle_events():
    events = []
    BacktestEngine(ExecutionConfig(), lambda event, payload: events.append(event)).run([intent()], candles())
    assert {"trade_intent", "execution", "exit", "run_completion"} <= set(events)
