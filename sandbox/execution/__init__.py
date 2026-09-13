"""Generic deterministic historical execution simulation."""

from sandbox.execution.engine import BacktestEngine, RunResult
from sandbox.execution.models import ExecutionConfig, TradeIntent

__all__ = ["BacktestEngine", "ExecutionConfig", "RunResult", "TradeIntent"]
