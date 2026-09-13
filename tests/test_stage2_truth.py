from datetime import datetime, timezone

import pandas as pd
import pytest

from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import CommissionModel, Direction, EntryType, ExecutionConfig, PositionStatus, SlippageModel, SpreadModel, TradeIntent

UTC = timezone.utc


def market(rows):
    return pd.DataFrame(rows, columns=["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]).assign(timestamp_utc=lambda x: pd.to_datetime(x.timestamp_utc, utc=True))


BASE = market([
    ("2024-01-01 00:00Z", 99, 101, 98, 100, 1, 0, 0),
    ("2024-01-01 00:01Z", 100, 105, 95, 102, 1, 0, 0),
    ("2024-01-01 00:02Z", 102, 121, 101, 120, 1, 0, 0),
])


def intent(direction=Direction.LONG, stop=90, target=120, ident="i1", signal="2024-01-01 00:00:30Z", entry=EntryType.MARKET_NEXT_OPEN):
    return TradeIntent(trade_intent_id=ident, strategy_id="manual-truth", experiment_id="truth", dataset_id="synthetic-v1", symbol="TEST", direction=direction, signal_timestamp=pd.Timestamp(signal).to_pydatetime(), requested_entry_type=entry, stop_loss=stop, take_profit=target)


def one(item, candles=BASE, config=None):
    return BacktestEngine(config or ExecutionConfig()).run([item], candles).ledger[0]


def test_long_clean_plus_2r_winner_after_several_bars():
    record = one(intent())
    assert record.exit_reason == "TAKE_PROFIT"
    assert record.gross_r == pytest.approx(2.0)
    assert record.bars_held == 2


def test_long_clean_minus_1r_loser():
    candles = BASE.copy(); candles.loc[1, ["low", "close"]] = [89, 91]
    record = one(intent(), candles)
    assert record.exit_reason == "STOP_LOSS" and record.gross_r == pytest.approx(-1.0)


def test_long_same_bar_is_conservative_loss():
    candles = BASE.copy(); candles.loc[1, ["high", "low"]] = [121, 89]
    record = one(intent(), candles)
    assert record.exit_reason == "SAME_BAR_AMBIGUOUS_SL_FIRST" and record.gross_r == pytest.approx(-1.0)


def test_long_gap_through_stop_executes_at_adverse_open():
    candles = BASE.copy(); candles.loc[2, ["open", "high", "low", "close"]] = [85, 88, 84, 86]
    record = one(intent(), candles)
    assert record.exit_reason == "GAP_THROUGH_SL" and record.exit_price == 85
    assert record.gross_r == pytest.approx(-1.5)


def test_short_clean_plus_2r_winner():
    candles = BASE.copy(); candles.loc[2, ["open", "high", "low", "close"]] = [98, 99, 79, 80]
    record = one(intent(Direction.SHORT, stop=110, target=80), candles)
    assert record.exit_reason == "TAKE_PROFIT" and record.gross_r == pytest.approx(2.0)


def test_short_clean_minus_1r_loser():
    candles = BASE.copy(); candles.loc[1, ["high", "close"]] = [111, 109]
    record = one(intent(Direction.SHORT, stop=110, target=80), candles)
    assert record.exit_reason == "STOP_LOSS" and record.gross_r == pytest.approx(-1.0)


def test_short_same_bar_is_conservative_loss():
    candles = BASE.copy(); candles.loc[1, ["high", "low"]] = [111, 79]
    record = one(intent(Direction.SHORT, stop=110, target=80), candles)
    assert record.exit_reason == "SAME_BAR_AMBIGUOUS_SL_FIRST" and record.gross_r == pytest.approx(-1.0)


def test_short_gap_through_stop_executes_at_adverse_open():
    candles = BASE.copy(); candles.loc[2, ["open", "high", "low", "close"]] = [115, 116, 114, 115]
    record = one(intent(Direction.SHORT, stop=110, target=80), candles)
    assert record.exit_reason == "GAP_THROUGH_SL" and record.gross_r == pytest.approx(-1.5)


def test_invalid_stop_is_preserved_as_rejection():
    record = one(intent(stop=101))
    assert record.status == PositionStatus.REJECTED and record.rejection_reason == "INVALID_STOP_LOSS"


def test_missing_next_open_is_rejected():
    record = one(intent(signal="2024-01-01 00:02:30Z"))
    assert record.status == PositionStatus.REJECTED and record.rejection_reason == "MISSING_NEXT_OPEN_EXECUTION_CANDLE"


@pytest.mark.parametrize("config, expected, field", [
    (ExecutionConfig(spread_model=SpreadModel.FIXED_SPREAD, fixed_spread_price=1), 1.9, "spread_cost"),
    (ExecutionConfig(commission_model=CommissionModel.FIXED_PER_TRADE, fixed_commission=2), 1.8, "commission_cost"),
    (ExecutionConfig(slippage_model=SlippageModel.FIXED_PRICE_SLIPPAGE, fixed_slippage_price=1), 1.9, "slippage_cost"),
    (ExecutionConfig(spread_model=SpreadModel.FIXED_SPREAD, fixed_spread_price=1, commission_model=CommissionModel.FIXED_PER_TRADE, fixed_commission=2), 1.7, "commission_cost"),
])
def test_costs_reduce_manual_2r_winner(config, expected, field):
    record = one(intent(), config=config)
    assert record.net_r == pytest.approx(expected)
    assert getattr(record, field) > 0


def test_historical_spread_is_points_not_pips():
    candles = BASE.copy(); candles["spread"] = 10
    config = ExecutionConfig(spread_model=SpreadModel.HISTORICAL_CANDLE_SPREAD, symbol_point=0.01)
    record = one(intent(), candles, config)
    assert record.spread_cost == pytest.approx(0.1)
    assert record.net_r == pytest.approx(1.99)
