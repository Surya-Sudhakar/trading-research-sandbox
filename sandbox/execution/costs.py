from __future__ import annotations

from sandbox.execution.models import CommissionModel, ExecutionConfig, SlippageModel, SpreadModel


def spread_distance(config: ExecutionConfig, candle_spread_points: float) -> float:
    if config.spread_model == SpreadModel.ZERO_SPREAD:
        return 0.0
    if config.spread_model == SpreadModel.FIXED_SPREAD:
        return config.fixed_spread_price
    return max(0.0, float(candle_spread_points)) * config.symbol_point


def spread_cost(config: ExecutionConfig, candle_spread_points: float, quantity: float) -> float:
    return spread_distance(config, candle_spread_points) * quantity


def commission_cost(config: ExecutionConfig, quantity: float) -> float:
    if config.commission_model == CommissionModel.ZERO_COMMISSION:
        return 0.0
    if config.commission_model == CommissionModel.FIXED_PER_TRADE:
        return config.fixed_commission
    return config.commission_per_lot_per_side * quantity * 2.0


def slippage_cost(config: ExecutionConfig, quantity: float) -> float:
    if config.slippage_model == SlippageModel.ZERO_SLIPPAGE:
        return 0.0
    return config.fixed_slippage_price * quantity

