from typing import Protocol
import pandas as pd
from .models import ProviderMetadata


class ProviderAdapter(Protocol):
    def read(self) -> pd.DataFrame: ...


class ExternalCSVAdapter:
    """Preserve original strings for audit; parsing/sanitation is separate."""
    def __init__(self, path, columns, timestamp_format, source_timezone):
        self.path=path;self.columns=columns
        self.timestamp_format=timestamp_format;self.source_timezone=source_timezone

    def read(self):
        raw=pd.read_csv(self.path,dtype=str,keep_default_na=False)
        result=raw.rename(columns={v:k for k,v in self.columns.items()})
        if self.timestamp_format == "UNIX_MILLISECONDS":
            numeric=pd.to_numeric(result["timestamp"],errors="coerce")
            valid=numeric.notna() & (numeric%1 == 0)
            result["timestamp_utc"]=pd.to_datetime(numeric.where(valid),unit="ms",utc=True,errors="coerce")
        elif self.timestamp_format == "ISO8601":
            def parse(value):
                try:
                    t=pd.Timestamp(value)
                    return (t.tz_localize(self.source_timezone,ambiguous="raise",nonexistent="raise")
                            if t.tzinfo is None else t).tz_convert("UTC")
                except (ValueError,TypeError):return pd.NaT
            result["timestamp_utc"]=pd.to_datetime(result["timestamp"].map(parse),utc=True)
        else:raise ValueError("unsupported timestamp format")
        return result


class MT5SourceAdapter:
    def __init__(self, adapter, symbol, start, end):
        self.adapter=adapter;self.symbol=symbol;self.start=start;self.end=end

    def read(self):
        return self.adapter.rates(self.symbol,self.start,self.end).copy()

