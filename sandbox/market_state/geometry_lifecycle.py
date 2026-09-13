"""Historical availability of completed projection geometry, without expiry."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re

from .block_aggregation import AnchoredBlockCandle
from .price_geometry import (
    CompletedBlockBodyProjection, build_completed_block_body_projection,
    build_big_brother_common_geometry,
)


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")


@dataclass(frozen=True)
class HistoricalGeometryInstance:
    geometry_family_id: str
    source_local_trading_date: date
    source_block_label: str
    source_block_index: int
    source_block_start_utc: datetime
    source_block_end_utc: datetime
    available_at_utc: datetime
    projection: CompletedBlockBodyProjection

    def __post_init__(self):
        if not isinstance(self.geometry_family_id, str) or re.fullmatch(r"[A-Z0-9_]+", self.geometry_family_id) is None:
            raise ValueError("invalid geometry_family_id")
        for stamp in (self.source_block_start_utc, self.source_block_end_utc, self.available_at_utc):
            _utc(stamp)
        if self.source_block_start_utc >= self.source_block_end_utc:
            raise ValueError("source window must have positive duration")
        if self.available_at_utc != self.source_block_end_utc:
            raise ValueError("availability must equal source block end")
        if not isinstance(self.projection, CompletedBlockBodyProjection):
            raise TypeError("projection must be CompletedBlockBodyProjection")
        for field in ("local_trading_date", "block_label", "block_index", "block_start_utc", "block_end_utc"):
            if getattr(self, "source_" + field) != getattr(self.projection, field):
                raise ValueError("inconsistent projection source metadata")


def _instance(candle, family, projection):
    return HistoricalGeometryInstance(
        family, candle.local_trading_date, candle.block_label, candle.block_index,
        candle.block_start_utc, candle.block_end_utc, candle.block_end_utc, projection)


def build_geometry_instance(candle: AnchoredBlockCandle, geometry_family_id: str,
                            level_specs) -> HistoricalGeometryInstance:
    if not isinstance(candle, AnchoredBlockCandle):
        raise TypeError("candle must be AnchoredBlockCandle")
    projection = build_completed_block_body_projection(candle, level_specs)
    return _instance(candle, geometry_family_id, projection)


BIG_BROTHER_GEOMETRY_FAMILY_ID = "BIG_BROTHER_COMMON"


def build_big_brother_geometry_instance(candle: AnchoredBlockCandle) -> HistoricalGeometryInstance:
    if not isinstance(candle, AnchoredBlockCandle):
        raise TypeError("candle must be AnchoredBlockCandle")
    return _instance(candle, BIG_BROTHER_GEOMETRY_FAMILY_ID,
                     build_big_brother_common_geometry(candle))


def _order(instance):
    return instance.available_at_utc, instance.source_block_start_utc


def build_historical_geometry_instances(candles, builder=build_big_brother_geometry_instance) -> tuple[HistoricalGeometryInstance, ...]:
    materialized = tuple(candles)
    previous = None
    windows = set()
    for candle in materialized:
        if not isinstance(candle, AnchoredBlockCandle):
            raise TypeError("candles must contain AnchoredBlockCandle")
        _utc(candle.block_start_utc)
        _utc(candle.block_end_utc)
        window = (candle.block_start_utc, candle.block_end_utc)
        if window in windows:
            raise ValueError("duplicate source window")
        windows.add(window)
        if previous is not None and candle.block_start_utc <= previous:
            raise ValueError("candles must be strictly ordered by block_start_utc")
        previous = candle.block_start_utc
    result = []
    for candle in materialized:
        instance = builder(candle)
        if not isinstance(instance, HistoricalGeometryInstance):
            raise TypeError("builder must return HistoricalGeometryInstance")
        for field in ("local_trading_date", "block_label", "block_index", "block_start_utc", "block_end_utc"):
            if getattr(instance, "source_" + field) != getattr(candle, field):
                raise ValueError("builder returned a different source candle")
        result.append(instance)
    return tuple(sorted(result, key=_order))


def geometry_available_at(instances, decision_time_utc) -> tuple[HistoricalGeometryInstance, ...]:
    _utc(decision_time_utc)
    available = []
    for instance in instances:
        if not isinstance(instance, HistoricalGeometryInstance):
            raise TypeError("instances must contain HistoricalGeometryInstance")
        if instance.available_at_utc <= decision_time_utc:
            available.append(instance)
    return tuple(sorted(available, key=_order))
