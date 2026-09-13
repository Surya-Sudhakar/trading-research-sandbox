from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable

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


def canonical_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"market data missing columns: {', '.join(missing)}")
    result = frame.copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    if result.timestamp_utc.duplicated().any():
        raise ValueError("market data contains duplicate timestamps")
    return result.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)


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
        duplicates = {x.trade_intent_id for x in items if sum(y.trade_intent_id == x.trade_intent_id for y in items) > 1}
        markets = {symbol: canonical_market_data(frame) for symbol, frame in candles.items()} if isinstance(candles, dict) else None
        ledger = []
        for intent in sorted(items, key=lambda x: (x.signal_timestamp, x.trade_intent_id)):
            self._emit("trade_intent", trade_intent_id=intent.trade_intent_id)
            frame = markets.get(intent.symbol) if markets is not None else canonical_market_data(candles)
            if intent.trade_intent_id in duplicates:
                record = self._reject(intent, "DUPLICATE_TRADE_INTENT_ID")
            elif frame is None:
                record = self._reject(intent, "SYMBOL_DATA_NOT_FOUND")
            else:
                record = self._execute(intent, frame)
            ledger.append(record)
        fingerprint = run_fingerprint(items, self.config)
        self._emit("run_completion", run_fingerprint=fingerprint, ledger_rows=len(ledger))
        return RunResult(fingerprint, tuple(ledger), self.config)

    def _base(self, intent: TradeIntent) -> dict:
        return dict(
            trade_id="trade-" + hashlib.sha256(intent.trade_intent_id.encode()).hexdigest()[:24],
            trade_intent_id=intent.trade_intent_id, strategy_id=intent.strategy_id,
            experiment_id=intent.experiment_id, dataset_id=intent.dataset_id,
            symbol=intent.symbol, direction=intent.direction,
            signal_timestamp=intent.signal_timestamp, stop_loss=intent.stop_loss,
            take_profit=intent.take_profit, quantity=intent.quantity,
            intent_metadata_json=json.dumps(intent.metadata, sort_keys=True, separators=(",", ":")),
        )

    def _reject(self, intent: TradeIntent, reason: str) -> LedgerRecord:
        self._emit("rejection", trade_intent_id=intent.trade_intent_id, reason=reason)
        return LedgerRecord(**self._base(intent), exit_reason="REJECTED", rejection_reason=reason, status=PositionStatus.REJECTED)

    def _execute(self, intent: TradeIntent, frame: pd.DataFrame) -> LedgerRecord:
        if intent.requested_entry_type in (EntryType.LIMIT, EntryType.STOP):
            return self._reject(intent, "UNSUPPORTED_ENTRY_TYPE")
        timestamps = frame.timestamp_utc
        signal = pd.Timestamp(intent.signal_timestamp)
        interval = pd.Timedelta(int(self.config.candle_interval_seconds) * 1_000_000_000, unit="ns")
        dataset_end = timestamps.max() + interval
        if signal < timestamps.min() or signal > dataset_end:
            return self._reject(intent, "SIGNAL_OUTSIDE_DATASET")
        if intent.requested_entry_type == EntryType.MARKET_NEXT_OPEN:
            candidates = frame.index[timestamps >= signal]
            if len(candidates) == 0:
                return self._reject(intent, "MISSING_NEXT_OPEN_EXECUTION_CANDLE")
            entry_index = int(candidates[0]); entry_price = float(frame.at[entry_index, "open"]); exit_start = entry_index
        else:
            close_times = timestamps + interval
            candidates = frame.index[close_times == signal]
            if len(candidates) == 0:
                return self._reject(intent, "IMPOSSIBLE_SIGNAL_TIMESTAMP")
            entry_index = int(candidates[0]); entry_price = float(frame.at[entry_index, "close"]); exit_start = entry_index + 1
        entry_timestamp = (signal if intent.requested_entry_type == EntryType.MARKET_CLOSE else timestamps.iloc[entry_index]).to_pydatetime()
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
        for index in range(exit_start, len(frame)):
            if index > entry_index:
                elapsed = pd.Timestamp(frame.at[index, "timestamp_utc"]) - pd.Timestamp(frame.at[index-1, "timestamp_utc"])
                if elapsed > interval:
                    market_gap_count += 1
                    self._emit("market_gap", trade_intent_id=intent.trade_intent_id, elapsed=str(elapsed), reopening_timestamp=pd.Timestamp(frame.at[index, "timestamp_utc"]).isoformat())
            outcome = self._resolve_bar(intent, frame.iloc[index])
            if outcome is not None:
                exit_price, reason = outcome
                direction_sign = 1.0 if intent.direction == Direction.LONG else -1.0
                gross = (exit_price - entry_price) * direction_sign * intent.quantity
                spread = spread_cost(self.config, float(frame.at[entry_index, "spread"]), intent.quantity)
                commission = commission_cost(self.config, intent.quantity)
                slippage = slippage_cost(self.config, intent.quantity)
                net = gross - spread - commission - slippage
                risk_value = risk * intent.quantity
                self._emit("exit", trade_intent_id=intent.trade_intent_id, reason=reason)
                return LedgerRecord(**self._base(intent), entry_timestamp=entry_timestamp, entry_price=entry_price,
                    exit_timestamp=pd.Timestamp(frame.at[index, "timestamp_utc"]).to_pydatetime(), exit_price=exit_price,
                    exit_reason=reason, gross_pnl=gross, spread_cost=spread, commission_cost=commission,
                    slippage_cost=slippage, net_pnl=net, gross_r=gross/risk_value, net_r=net/risk_value,
                    bars_held=index-entry_index+1, market_gap_count=market_gap_count, initial_risk_price_distance=risk, status=PositionStatus.CLOSED)
        return LedgerRecord(**self._base(intent), entry_timestamp=entry_timestamp, entry_price=entry_price,
            exit_reason="END_OF_DATA", initial_risk_price_distance=risk,
            bars_held=max(0, len(frame)-entry_index), market_gap_count=market_gap_count, status=PositionStatus.OPEN)

    def _resolve_bar(self, intent: TradeIntent, row: pd.Series) -> tuple[float, str] | None:
        open_, high, low = float(row.open), float(row.high), float(row.low)
        spread = spread_distance(self.config, float(row.spread))
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
