"""Deterministic linear completed-block body geometry, without trading logic."""
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
import re

from .block_aggregation import AnchoredBlockCandle


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite numeric value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("expected a finite numeric value") from exc
    if not isfinite(result):
        raise ValueError("expected a finite numeric value")
    return result


def _level_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Z0-9_]+", value) is None:
        raise ValueError("invalid level_id")


@dataclass(frozen=True)
class ProjectionLevelSpec:
    level_id: str
    coordinate: float

    def __post_init__(self):
        _level_id(self.level_id)
        object.__setattr__(self, "coordinate", _finite(self.coordinate))


@dataclass(frozen=True)
class ProjectedLevel:
    level_id: str
    coordinate: float
    price: float

    def __post_init__(self):
        _level_id(self.level_id)
        object.__setattr__(self, "coordinate", _finite(self.coordinate))
        object.__setattr__(self, "price", _finite(self.price))


class BodyDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    FLAT = "FLAT"


@dataclass(frozen=True)
class CompletedBlockBodyProjection:
    local_trading_date: date
    block_label: str
    block_index: int
    block_start_utc: datetime
    block_end_utc: datetime
    open: float
    close: float
    signed_body_move: float
    body_size: float
    direction: BodyDirection
    anchor_zero_price: float
    anchor_minus_one_price: float
    levels: tuple[ProjectedLevel, ...]

    def __post_init__(self):
        object.__setattr__(self, "levels", tuple(self.levels))


def projection_price(anchor_zero_price, anchor_minus_one_price, coordinate) -> float:
    zero = _finite(anchor_zero_price)
    minus_one = _finite(anchor_minus_one_price)
    coordinate = _finite(coordinate)
    return _finite(zero + coordinate * (zero - minus_one))


def build_completed_block_body_projection(candle: AnchoredBlockCandle,
                                          level_specs) -> CompletedBlockBodyProjection:
    if not isinstance(candle, AnchoredBlockCandle):
        raise TypeError("candle must be AnchoredBlockCandle")
    specs = tuple(level_specs)
    if any(not isinstance(spec, ProjectionLevelSpec) for spec in specs):
        raise TypeError("level_specs must contain ProjectionLevelSpec")
    if len({spec.level_id for spec in specs}) != len(specs):
        raise ValueError("duplicate level IDs")
    if len({spec.coordinate for spec in specs}) != len(specs):
        raise ValueError("duplicate coordinates")
    opening, closing = _finite(candle.open), _finite(candle.close)
    move = _finite(closing - opening)
    direction = BodyDirection.BULLISH if closing > opening else (
        BodyDirection.BEARISH if closing < opening else BodyDirection.FLAT)
    levels = tuple(ProjectedLevel(spec.level_id, spec.coordinate,
                                  projection_price(opening, closing, spec.coordinate)) for spec in specs)
    return CompletedBlockBodyProjection(
        candle.local_trading_date, candle.block_label, candle.block_index,
        candle.block_start_utc, candle.block_end_utc, opening, closing,
        move, abs(move), direction, opening, closing, levels)


BIG_BROTHER_COMMON_LEVELS = (
    ProjectionLevelSpec("NEG_3_0", -3.0),
    ProjectionLevelSpec("NEG_3_3", -3.3),
    ProjectionLevelSpec("NEG_3_5", -3.5),
)


def build_big_brother_common_geometry(candle: AnchoredBlockCandle) -> CompletedBlockBodyProjection:
    return build_completed_block_body_projection(candle, BIG_BROTHER_COMMON_LEVELS)
