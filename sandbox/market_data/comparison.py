import numpy as np
import pandas as pd


def compare_providers(a,b,*,pip_size=0.0001):
    if pip_size<=0:raise ValueError("positive pip size required")
    for f in (a,b):
        ts=pd.DatetimeIndex(f.timestamp_utc)
        if ts.tz is None or ts.has_duplicates:raise ValueError("aware unique timestamps required")
        if not np.isfinite(f[["open","high","low","close"]].to_numpy(dtype=float)).all():
            raise ValueError("finite prices required")
    if "symbol" in a and "symbol" in b and set(a.symbol)!=set(b.symbol):raise ValueError("symbol mismatch")
    if "timeframe" in a and "timeframe" in b and set(a.timeframe)!=set(b.timeframe):raise ValueError("timeframe mismatch")
    cols=["timestamp_utc","open","high","low","close"]
    joined=a[cols].merge(b[cols],on="timestamp_utc",how="outer",suffixes=("_a","_b"),indicator=True,validate="one_to_one")
    matched=joined[joined._merge.eq("both")].copy()
    for col in cols[1:]:
        matched[col+"_difference"]=matched[col+"_a"]-matched[col+"_b"]
        matched[col+"_absolute_difference"]=matched[col+"_difference"].abs()
        matched[col+"_difference_pips"]=matched[col+"_difference"]/pip_size
    matched["direction_agreement"]=np.sign(matched.close_a-matched.open_a)==np.sign(matched.close_b-matched.open_b)
    matched["range_difference"]=(matched.high_a-matched.low_a)-(matched.high_b-matched.low_b)
    close=matched.close_absolute_difference
    ar=matched.high_a-matched.low_a;br=matched.high_b-matched.low_b
    correlation=float(ar.corr(br)) if len(matched)>1 and ar.std()>0 and br.std()>0 else None
    return matched,dict(matched_bars=len(matched),missing_on_a=int(joined._merge.eq("right_only").sum()),
                        missing_on_b=int(joined._merge.eq("left_only").sum()),
                        median_absolute_close_difference=float(close.median()) if len(close) else None,
                        p95_absolute_close_difference=float(close.quantile(.95)) if len(close) else None,
                        maximum_absolute_close_difference=float(close.max()) if len(close) else None,
                        direction_agreement_rate=float(matched.direction_agreement.mean()) if len(matched) else None,
                        range_correlation=correlation,pip_size=pip_size,
                        median_absolute_close_difference_pips=float(close.median()/pip_size) if len(close) else None)

