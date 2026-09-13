"""Safe JSON snapshots of the runner's fixed set of frozen research objects.

Object references preserve the identity requirements of session-context models.
The decoder never imports a module or executes a callable named by an artifact.
"""
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from importlib import import_module
from numbers import Integral
import pickle
from zoneinfo import ZoneInfo
import pytz

from .canonical import canonical_json, sha256_canonical


_MODULES = (
    "market_state.block_aggregation", "market_state.session_clock",
    "market_state.price_geometry", "market_state.research_anchor", "market_state.future_outcome",
    "market_state.multitimeframe_context", "market_state.research_multitimeframe_context",
    "market_state.session_summary", "market_state.session_relationship",
    "market_state.research_session_context", "market_state.research_session_relationship",
    "market_state.discovery_market_context", "market_state.swings", "market_state.touch_transition",
    "market_state.research_record", "market_state.discovery_projection", "market_state.discovery_dataset",
    "research.discovery_record_builder", "research.ml_discovery_matrix", "research.ml_preprocessing",
    "research.ml_lightgbm", "research.ml_evaluation", "research.ml_clue_extraction", "research.ai_researcher",
)


def _types():
    allowed = {}
    for name in _MODULES:
        module = import_module("sandbox." + name)
        for cls in vars(module).values():
            if (isinstance(cls, type) and cls.__module__ == module.__name__
                    and (is_dataclass(cls) or issubclass(cls, Enum))):
                allowed[cls.__module__ + "." + cls.__name__] = cls
    return allowed


def pack(value):
    memo = {}
    def visit(item):
        if isinstance(item, Enum):
            return {"enum": type(item).__module__ + "." + type(item).__name__, "value": item.value}
        if is_dataclass(item):
            if id(item) in memo:
                return {"ref": memo[id(item)]}
            number = len(memo)
            memo[id(item)] = number
            return {"id": number, "type": type(item).__module__ + "." + type(item).__name__,
                    "fields": {f.name: visit(getattr(item, f.name)) for f in fields(item)}}
        if isinstance(item, datetime):
            return {"datetime": item.isoformat(), "zone": getattr(item.tzinfo, "key", None), "fold": item.fold}
        if isinstance(item, date):
            return {"date": item.isoformat()}
        if isinstance(item, time):
            return {"time": item.isoformat(), "fold": item.fold}
        if isinstance(item, tuple):
            return {"tuple": [visit(x) for x in item]}
        if isinstance(item, dict):
            return {"mapping": [[k, visit(v)] for k, v in sorted(item.items())]}
        if isinstance(item, Integral) and not isinstance(item, bool):
            return int(item)
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise TypeError(f"unsupported snapshot type: {type(item)}")
    return visit(value)


def unpack(value):
    allowed, memo = _types(), {}
    def visit(item):
        if not isinstance(item, dict):
            return item
        if "ref" in item:
            return memo[item["ref"]]
        if "type" in item:
            cls = allowed[item["type"]]
            # Decode in declaration order, independent of sorted JSON key order.
            obj = cls(**{f.name: visit(item["fields"][f.name]) for f in fields(cls)})
            memo[item["id"]] = obj
            return obj
        if "enum" in item:
            return allowed[item["enum"]](item["value"])
        if "datetime" in item:
            stamp = datetime.fromisoformat(item["datetime"])
            if item["zone"]:
                stamp = stamp.astimezone(ZoneInfo(item["zone"]))
            return stamp.replace(fold=item["fold"])
        if "date" in item:
            return date.fromisoformat(item["date"])
        if "time" in item:
            return time.fromisoformat(item["time"]).replace(fold=item["fold"])
        if "tuple" in item:
            return tuple(visit(x) for x in item["tuple"])
        if "mapping" in item:
            return {k: visit(v) for k, v in item["mapping"]}
        raise ValueError("invalid snapshot node")
    return visit(value)


def identity(value):
    return sha256_canonical(pack(value))


def snapshot_json(value):
    return canonical_json(pack(value))


class _RestrictedUnpickler(pickle.Unpickler):
    """Load only the runner's fixed frozen value types from local snapshots."""

    def find_class(self, module, name):
        allowed = _types()
        qualified = module + "." + name
        if qualified in allowed:
            return allowed[qualified]
        safe = {
            "datetime.date": date,
            "datetime.datetime": datetime,
            "datetime.time": time,
            "datetime.timedelta": timedelta,
            "datetime.timezone": timezone,
            "zoneinfo.ZoneInfo": ZoneInfo,
            "pytz._UTC": type(pytz.UTC),
        }
        if qualified in safe:
            return safe[qualified]
        raise pickle.UnpicklingError(f"forbidden snapshot global: {qualified}")


class _SnapshotPickler(pickle.Pickler):
    def reducer_override(self, value):
        if type(value) is ZoneInfo:
            return ZoneInfo, (value.key,)
        return NotImplemented


def write_binary_snapshot(path, value):
    """Stream a compact snapshot without materializing a second object graph."""
    with open(path, "wb") as stream:
        _SnapshotPickler(stream, protocol=5).dump(value)


def read_binary_snapshot(path):
    with open(path, "rb") as stream:
        return _RestrictedUnpickler(stream).load()
