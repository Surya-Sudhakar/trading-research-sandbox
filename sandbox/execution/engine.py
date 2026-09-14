from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from sandbox import EXECUTION_ENGINE_VERSION, LEDGER_SCHEMA_VERSION, METRICS_VERSION
from sandbox.execution.costs import commission_cost, slippage_cost, spread_cost, spread_distance
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, LedgerRecord, PositionStatus, TradeIntent

REQUIRED = ["timestamp_utc", "open", "high", "low", "close", "spread"]
AuditHook = Callable[[str, dict], None]


@dataclass(frozen=True)
class RunResult:
    run_fingerprint: str
    ledger: tuple[LedgerRecord, ...]
    execution_config: ExecutionConfig
    execution_engine_version: str = EXECUTION_ENGINE_VERSION
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    metrics_version: str = METRICS_VERSION

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.model_dump(mode="json") for record in self.ledger])


@dataclass(frozen=True)
class _PreparedMarket:
    timestamps: pd.DatetimeIndex
    timestamp_ns: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    spreads: np.ndarray
    gap_after_previous: np.ndarray
    interval: pd.Timedelta
    interval_ns: int
    start: pd.Timestamp | None
    dataset_end: pd.Timestamp | None


def canonical_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"market data missing columns: {', '.join(missing)}")
    result = frame.copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    if result.timestamp_utc.duplicated().any():
        raise ValueError("market data contains duplicate timestamps")
    return result.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)


def _prepare_market(frame: pd.DataFrame, interval: pd.Timedelta) -> _PreparedMarket:
    canonical = canonical_market_data(frame)
    timestamps = pd.DatetimeIndex(canonical["timestamp_utc"])
    # Pandas 3 may preserve datetime64[us] resolution. Explicitly normalize the
    # integer search/gap representation to nanoseconds because Timestamp.value
    # and Timedelta.value are nanosecond-based.
    timestamp_ns = timestamps.as_unit("ns").asi8
    interval_ns = int(interval.value)
    gaps = np.zeros(len(timestamps), dtype=np.bool_)
    if len(timestamps) > 1:
        gaps[1:] = (timestamp_ns[1:] - timestamp_ns[:-1]) > interval_ns
    start = timestamps[0] if len(timestamps) else None
    dataset_end = timestamps[-1] + interval if len(timestamps) else None
    return _PreparedMarket(
        timestamps=timestamps,
        timestamp_ns=timestamp_ns,
        opens=canonical["open"].to_numpy(dtype=np.float64, copy=False),
        highs=canonical["high"].to_numpy(dtype=np.float64, copy=False),
        lows=canonical["low"].to_numpy(dtype=np.float64, copy=False),
        closes=canonical["close"].to_numpy(dtype=np.float64, copy=False),
        spreads=canonical["spread"].to_numpy(dtype=np.float64, copy=False),
        gap_after_previous=gaps,
        interval=interval,
        interval_ns=interval_ns,
        start=start,
        dataset_end=dataset_end,
    )


def run_fingerprint(intents: Iterable[TradeIntent], config: ExecutionConfig) -> str:
    intents = list(intents)
    dataset_ids = sorted({item.dataset_id for item in intents})
    payload = {
        "dataset_ids": dataset_ids,
        "trade_intents": [item.model_dump(mode="json") for item in sorted(intents, key=lambda x: (x.trade_intent_id, x.signal_timestamp))],
        "execution_config": config.model_dump(mode="json"),
        "execution_engine_version": EXECUTION_ENGINE_VERSION,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class BacktestEngine:
    def __init__(self, config: ExecutionConfig, audit_hook: AuditHook | None = None):
        self.config = config
        self.audit_hook = audit_hook or (lambda _event, _payload: None)

    def _emit(self, event: str, **payload) -> None:
        self.audit_hook(event, payload)

    def run(self, intents: Iterable[TradeIntent], candles: pd.DataFrame | dict[str, pd.DataFrame]) -> RunResult:
        items = list(intents)
        counts = Counter(item.trade_intent_id for item in items)
        duplicates = {trade_intent_id for trade_intent_id, count in counts.items() if count > 1}
        interval = pd.Timedelta(int(self.config.candle_interval_seconds) * 1_000_000_000, unit="ns")
        if isinstance(candles, dict):
            markets = {symbol: _prepare_market(frame, interval) for symbol, frame in candles.items()}
            single_market = None
        else:
            markets = None
            single_market = _prepare_market(candles, interval)
        ledger = []
        for intent in sorted(items, key=lambda x: (x.signal_timestamp, x.trade_intent_id)):
            self._emit("trade_intent", trade_intent_id=intent.trade_intent_id)
            market = markets.get(intent.symbol) if markets is not None else single_market
            if intent.trade_intent_id in duplicates:
                record = self._reject(intent, "DUPLICATE_TRADE_INTENT_ID")
            elif market is None:
                record = self._reject(intent, "SYMBOL_DATA_NOT_FOUND")
            else:
                record = self._execute(intent, market)
            ledger.append(record)
        fingerprint = run_fingerprint(items, self.config)
        self._emit("run_completion", run_fingerprint=fingerprint, ledger_rows=len(ledger))
        return RunResult(fingerprint, tuple(ledger), self.config)

    def _base(self, intent: TradeIntent) -> dict:
        return dict(
            trade_id="trade-" + hashlib.sha256(intent.trade_intent_id.encode()).hexdigest()[:24],
            trade_intent_id=intent.trade_intent_id,
            strategy_id=intent.strategy_id,
            experiment_id=intent.experiment_id,
            dataset_id=intent.dataset_id,
            symbol=intent.symbol,
            direction=intent.direction,
            signal_timestamp=intent.signal_timestamp,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            quantity=intent.quantity,
            intent_metadata_json=json.dumps(intent.metadata, sort_keys=True, separators=(",", ":")),
        )

    def _reject(self, intent: TradeIntent, reason: str) -> LedgerRecord:
        self._emit("rejection", trade_intent_id=intent.trade_intent_id, reason=reason)
        return LedgerRecord(**self._base(intent), exit_reason="REJECTED", rejection_reason=reason, status=PositionStatus.REJECTED)

    def _entry_index(self, intent: TradeIntent, market: _PreparedMarket, signal: pd.Timestamp) -> int | None:
        if intent.requested_entry_type == EntryType.MARKET_NEXT_OPEN:
            position = int(np.searchsorted(market.timestamp_ns, signal.value, side="left"))
            return None if position >= len(market.timestamp_ns) else position
        open_ns = signal.value - market.interval_ns
        position = int(np.searchsorted(market.timestamp_ns, open_ns, side="left"))
        if position >= len(market.timestamp_ns) or int(market.timestamp_ns[position]) != open_ns:
            return None
        return position

    def _execute(self, intent: TradeIntent, market: _PreparedMarket) -> LedgerRecord:
        if intent.requested_entry_type in (EntryType.LIMIT, EntryType.STOP):
            return self._reject(intent, "UNSUPPORTED_ENTRY_TYPE")
        signal = pd.Timestamp(intent.signal_timestamp)
        if market.start is not None and (signal < market.start or signal > market.dataset_end):
            return self._reject(intent, "SIGNAL_OUTSIDE_DATASET")
        entry_index = self._entry_index(intent, market, signal)
        if entry_index is None:
            reason = ("MISSING_NEXT_OPEN_EXECUTION_CANDLE" if intent.requested_entry_type == EntryType.MARKET_NEXT_OPEN else "IMPOSSIBLE_SIGNAL_TIMESTAMP")
            return self._reject(intent, reason)
        if intent.requested_entry_type == EntryType.MARKET_NEXT_OPEN:
            entry_price = float(market.opens[entry_index])
            exit_start = entry_index
            entry_timestamp = market.timestamps[entry_index].to_pydatetime()
        else:
            entry_price = float(market.closes[entry_index])
            exit_start = entry_index + 1
            entry_timestamp = signal.to_pydatetime()
        if entry_timestamp < intent.signal_timestamp:
            return self._reject(intent, "EXECUTION_BEFORE_SIGNAL")
        if intent.direction == Direction.LONG:
            risk = entry_price - intent.stop_loss
            target_valid = intent.take_profit > entry_price
        else:
            risk = intent.stop_loss - entry_price
            target_valid = intent.take_profit < entry_price
        if risk <= 0:
            return self._reject(intent, "INVALID_STOP_LOSS")
        if not target_valid:
            return self._reject(intent, "INVALID_TAKE_PROFIT")
        self._emit("execution", trade_intent_id=intent.trade_intent_id, entry_timestamp=entry_timestamp.isoformat(), entry_price=entry_price)
        market_gap_count = 0
        length = len(market.timestamp_ns)
        for index in range(exit_start, length):
            if index > entry_index and market.gap_after_previous[index]:
                market_gap_count += 1
                elapsed = market.timestamps[index] - market.timestamps[index - 1]
                self._emit("market_gap", trade_intent_id=intent.trade_intent_id, elapsed=str(elapsed), reopening_timestamp=market.timestamps[index].isoformat())
            outcome = self._resolve_bar_values(intent, float(market.opens[index]), float(market.highs[index]), float(market.lows[index]), float(market.spreads[index]))
            if outcome is not None:
                exit_price, reason = outcome
                direction_sign = 1.0 if intent.direction == Direction.LONG else -1.0
                gross = (exit_price - entry_price) * direction_sign * intent.quantity
                spread = spread_cost(self.config, float(market.spreads[entry_index]), intent.quantity)
                commission = commission_cost(self.config, intent.quantity)
                slippage = slippage_cost(self.config, intent.quantity)
                net = gross - spread - commission - slippage
                risk_value = risk * intent.quantity
                self._emit("exit", trade_intent_id=intent.trade_intent_id, reason=reason)
                return LedgerRecord(**self._base(intent), entry_timestamp=entry_timestamp, entry_price=entry_price,
                    exit_timestamp=market.timestamps[index].to_pydatetime(), exit_price=exit_price,
                    exit_reason=reason, gross_pnl=gross, spread_cost=spread, commission_cost=commission,
                    slippage_cost=slippage, net_pnl=net, gross_r=gross/risk_value, net_r=net/risk_value,
                    bars_held=index-entry_index+1, market_gap_count=market_gap_count,
                    initial_risk_price_distance=risk, status=PositionStatus.CLOSED)
        return LedgerRecord(**self._base(intent), entry_timestamp=entry_timestamp, entry_price=entry_price,
            exit_reason="END_OF_DATA", initial_risk_price_distance=risk,
            bars_held=max(0, length-entry_index), market_gap_count=market_gap_count, status=PositionStatus.OPEN)

    def _resolve_bar_values(self, intent: TradeIntent, open_: float, high: float, low: float, candle_spread: float) -> tuple[float, str] | None:
        spread = spread_distance(self.config, candle_spread)
        if intent.direction == Direction.LONG:
            if open_ <= intent.stop_loss:
                return open_, "GAP_THROUGH_SL"
            if open_ >= intent.take_profit:
                return intent.take_profit, "GAP_BEYOND_TP_CONSERVATIVE"
            sl_hit, tp_hit = low <= intent.stop_loss, high >= intent.take_profit
        else:
            ask_open, ask_high, ask_low = open_ + spread, high + spread, low + spread
            if ask_open >= intent.stop_loss:
                return ask_open, "GAP_THROUGH_SL"
            if ask_open <= intent.take_profit:
                return intent.take_profit, "GAP_BEYOND_TP_CONSERVATIVE"
            sl_hit, tp_hit = ask_high >= intent.stop_loss, ask_low <= intent.take_profit
        if sl_hit and tp_hit:
            return intent.stop_loss, "SAME_BAR_AMBIGUOUS_SL_FIRST"
        if sl_hit:
            return intent.stop_loss, "STOP_LOSS"
        if tp_hit:
            return intent.take_profit, "TAKE_PROFIT"
        return None

    def _resolve_bar(self, intent: TradeIntent, row: pd.Series) -> tuple[float, str] | None:
        return self._resolve_bar_values(intent, float(row.open), float(row.high), float(row.low), float(row.spread))
