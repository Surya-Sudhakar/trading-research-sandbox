import pandas as pd

from sandbox.integrity import audit


def frame(times):
    return pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True), "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "tick_volume": 1, "spread": 1, "real_volume": 0})


def test_consecutive_m1_bars_are_clean():
    report = audit(frame(["2024-01-01 00:00Z", "2024-01-01 00:01Z"]))
    assert report.status == "clean"
    assert report.issues == ()


def test_duplicates_and_order_detected():
    report = audit(frame(["2024-01-01 00:01Z", "2024-01-01 00:00Z", "2024-01-01 00:00Z"]))
    codes = {x.code for x in report.issues}
    assert {"duplicate_timestamp", "out_of_order"} <= codes
    assert report.status == "error"


def test_one_missing_m1_bar_is_recorded_as_warning():
    report = audit(frame(["2024-01-01 00:00Z", "2024-01-01 00:02Z"]))
    assert report.status == "warning"
    assert report.issues[0].category == "suspicious_gap"
    assert report.issues[0].code == "missing_interval"


def test_large_weekday_gap_is_suspicious():
    report = audit(frame(["2024-01-01 00:00Z", "2024-01-01 00:10Z"]))
    assert report.issues[0].category == "suspicious_gap"


def test_weekend_gap_is_expected_closure():
    report = audit(frame(["2026-07-10 23:56Z", "2026-07-13 00:00Z"]))
    assert report.issues[0].category == "expected_market_closure"
    assert report.status == "clean"


def test_invalid_ohlc_remains_a_confirmed_error():
    bars = frame(["2024-01-01 00:00Z"])
    bars.loc[0, "high"] = 0.9
    report = audit(bars)
    assert report.status == "error"
    assert report.issues[0].category == "confirmed_data_error"
    assert report.issues[0].code == "invalid_ohlc"


def test_multiweek_outage_is_not_an_expected_weekend():
    report = audit(frame(["2026-07-10 23:56Z", "2026-07-20 00:00Z"]))
    assert report.status == "warning"
    assert report.issues[0].category == "suspicious_gap"
