from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def calendar_values(value: datetime, timezone_name: str) -> dict:
    utc=value.astimezone(timezone.utc);local=utc.astimezone(ZoneInfo(timezone_name))
    return {"calendar.utc_timestamp":utc.isoformat(),"calendar.new_york_timestamp":local.isoformat(),
        "calendar.utc_hour":utc.hour,"calendar.utc_minute":utc.minute,"calendar.day_of_week":utc.weekday(),
        "calendar.month":utc.month,"calendar.new_york_hour":local.hour,"calendar.new_york_minute":local.minute}


def time_block_values(value: datetime, blocks: tuple[int,...]) -> dict[str,float|int]:
    utc=value.astimezone(timezone.utc);minute=utc.hour*60+utc.minute+utc.second/60
    result={}
    for size in blocks:
        elapsed=minute%size
        result[f"calendar.block_{size}m_elapsed_minutes"]=elapsed
        result[f"calendar.block_{size}m_elapsed_fraction"]=elapsed/size
    return result
