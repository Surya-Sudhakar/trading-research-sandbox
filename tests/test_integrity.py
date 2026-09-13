import pandas as pd

from sandbox.integrity import audit


def frame(times):
    return pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True), "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "tick_volume": 1, "spread": 1, "real_volume": 0})


def test_duplicates_and_order_detected():
    report = audit(frame(["2024-01-01 00:01Z", "2024-01-01 00:00Z", "2024-01-01 00:00Z"]))
    codes = {x.code for x in report.issues}
    assert {"duplicate_timestamp", "out_of_order"} <= codes
    assert report.status == "error"


def test_short_missing_interval_is_confirmed_error():
    report = audit(frame(["2024-01-01 00:00Z", "2024-01-01 00:02Z"]))
    assert report.issues[0].category == "confirmed_data_error"


def test_large_weekday_gap_is_suspicious():
    report = audit(frame(["2024-01-01 00:00Z", "2024-01-01 00:10Z"]))
    assert report.issues[0].category == "suspicious_gap"


def test_weekend_gap_is_expected_closure():
    report = audit(frame(["2024-01-05 21:59Z", "2024-01-07 22:01Z"]))
    assert report.issues[0].category == "expected_market_closure"
    assert report.status == "clean"

