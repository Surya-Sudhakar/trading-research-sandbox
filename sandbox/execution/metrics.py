from __future__ import annotations

import math

import pandas as pd

from sandbox.execution.models import PositionStatus


def _drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values.fillna(0.0).cumsum()
    peaks = equity.cummax().clip(lower=0.0)
    return float((peaks - equity).max())


def _streak(signs: list[int], wanted: int) -> int:
    longest = current = 0
    for sign in signs:
        current = current + 1 if sign == wanted else 0
        longest = max(longest, current)
    return longest


def summarize(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return _summary_values(ledger, 0)
    closed = ledger[ledger.status == PositionStatus.CLOSED.value].copy()
    if not closed.empty:
        closed = closed.sort_values(["exit_timestamp", "trade_id"], kind="mergesort")
    return _summary_values(closed, len(ledger), int((ledger.status == PositionStatus.REJECTED.value).sum()))


def _summary_values(closed: pd.DataFrame, total: int, rejected: int = 0) -> dict:
    values = pd.to_numeric(closed.get("net_r", pd.Series(dtype=float)), errors="coerce").dropna()
    gross = pd.to_numeric(closed.get("gross_r", pd.Series(dtype=float)), errors="coerce").dropna()
    wins, losses = int((values > 0).sum()), int((values < 0).sum())
    breakeven = int((values == 0).sum())
    resolved = len(values)
    positive, negative = float(values[values > 0].sum()), float(values[values < 0].sum())
    profit_factor = positive / abs(negative) if negative < 0 else (None if positive > 0 else 0.0)
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in values]
    if closed.empty or "exit_timestamp" not in closed:
        years = 0.0
    else:
        dates = pd.to_datetime(closed.exit_timestamp, utc=True).dropna()
        span_days = max(1.0, (dates.max()-dates.min()).total_seconds()/86400) if len(dates) > 1 else 365.2425
        years = span_days / 365.2425
    return {
        "total_trade_intents": int(total), "executed_trades": int(total-rejected), "rejected_trades": int(rejected),
        "resolved_trades": int(resolved), "wins": wins, "losses": losses, "breakeven": breakeven,
        "win_rate": wins/resolved if resolved else 0.0, "loss_rate": losses/resolved if resolved else 0.0,
        "gross_R": float(gross.sum()), "net_R": float(values.sum()),
        "average_R": float(values.mean()) if resolved else 0.0, "median_R": float(values.median()) if resolved else 0.0,
        "profit_factor": profit_factor, "maximum_drawdown_R": _drawdown(values),
        "longest_winning_streak": _streak(signs, 1), "longest_losing_streak": _streak(signs, -1),
        "average_bars_held": float(pd.to_numeric(closed.bars_held).mean()) if resolved else 0.0,
        "trades_per_year": resolved/years if years else 0.0,
    }


def yearly_breakdown(ledger: pd.DataFrame) -> list[dict]:
    if ledger.empty or "status" not in ledger:
        return []
    closed = ledger[ledger.status == PositionStatus.CLOSED.value].copy()
    if closed.empty:
        return []
    closed["year"] = pd.to_datetime(closed.exit_timestamp, utc=True).dt.year
    result = []
    for year, group in closed.groupby("year", sort=True):
        summary = _summary_values(group, len(group))
        result.append({"year": int(year), "trades": len(group), "wins": summary["wins"], "losses": summary["losses"], "win_rate": summary["win_rate"], "net_R": summary["net_R"], "profit_factor": summary["profit_factor"], "max_drawdown_R": summary["maximum_drawdown_R"]})
    return result


def symbol_breakdown(ledger: pd.DataFrame) -> list[dict]:
    if ledger.empty or "symbol" not in ledger:
        return []
    result = []
    for symbol, group in ledger.groupby("symbol", sort=True):
        result.append({"symbol": symbol, **summarize(group)})
    return result
