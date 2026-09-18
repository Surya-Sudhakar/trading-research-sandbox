import json
from datetime import timedelta, timezone
import pandas as pd
import pytest
from sandbox.models import Candle
from sandbox.market_state.models import FeatureConfiguration
from sandbox.market_state.volatility import wilder_atr
from sandbox.market_state.universal import UniversalFeatureEngine, FeatureRow
from sandbox.market_state.universal.candle import geometry
from sandbox.market_state.universal.volatility import atr_series, percentile_series
from sandbox.market_state.universal.structure import prior_structure


def bars(n=160, start="2024-01-01", freq="15min"):
    base = [100+i*.1 for i in range(n)]
    return pd.DataFrame(dict(timestamp_utc=pd.date_range(start, periods=n, freq=freq, tz="UTC"),
        open=base, high=[x+.4 for x in base], low=[x-.2 for x in base],
        close=[x+.1 for x in base]))


def compute(frame, **kwargs):
    return UniversalFeatureEngine().compute(frame, symbol="EURUSD", **kwargs)


@pytest.mark.parametrize("ohlc,expected", [
    ((10, 15, 8, 14), (7, 4, 4/7, 1, 2, 12, 6/7, 1)),
    ((14, 15, 8, 10), (7, 4, 4/7, 1, 2, 12, 2/7, -1)),
    ((10, 15, 8, 10), (7, 0, 0, 5, 2, 10, 2/7, 0)),
    ((10, 10, 10, 10), (0, 0, None, 0, 0, 10, None, 0)),
])
def test_geometry(ohlc, expected):
    assert tuple(geometry(*ohlc).values()) == expected


def test_atr_independent_seed_recurrence_and_existing_semantics():
    # Constant closes: TRs are exactly 1..17; first bar deliberately ignored.
    h = [100] + [100+i/2 for i in range(1, 18)]
    l = [100] + [100-i/2 for i in range(1, 18)]
    c = [100]*18
    actual = atr_series(h, l, c)
    expected = sum(range(1, 15))/14
    assert actual[:14] == [None]*14
    assert actual[14] == expected
    for i in range(15, 18):
        expected = (13*expected+i)/14
        assert actual[i] == expected
    assert actual == [wilder_atr(h[:i], l[:i], c[:i], 14) for i in range(1, 19)]
    assert atr_series([1]*20, [1]*20, [1]*20) == [None]*20


def test_percentile_full_window_inclusive_ties():
    assert percentile_series([1, 2, 2, 0], 3) == [None, None, 100, 100/3]
    assert percentile_series([None, 1, 2], 3) == [None]*3


@pytest.mark.parametrize("close,high_break,low_break", [
    (11, True, False), (10, False, False), (5, False, False), (4, False, True)
])
def test_structure_excludes_current_and_strict_breakouts(close, high_break, low_break):
    values = prior_structure([10]*12+[999], [5]*12+[1], close, 12)
    assert values == dict(prior_high_12=10, prior_low_12=5,
                         breakout_high_12=high_break, breakout_low_12=low_break)


@pytest.mark.parametrize("tf,count,name", [
    ("H1", 4, "completed_h1_direction"), ("H3", 12, "completed_h3_direction")
])
def test_context_completion_exact_boundary_and_previous(tf, count, name):
    frame = bars(count*2)
    # Second aggregate is bearish; incomplete prefixes must keep first direction.
    frame.loc[count:, "open"] = 200
    frame.loc[count:, "high"] = 201
    frame.loc[count:, "low"] = 80
    frame.loc[count:, "close"] = 90
    rows = compute(frame)
    assert all(getattr(r, name) is None for r in rows[:count-1])
    assert getattr(rows[count-1], name) == 1
    assert all(getattr(r, name) == 1 for r in rows[count:2*count-1])
    assert getattr(rows[-1], name) == -1


def test_future_mutation_and_prefix_equivalence_every_feature():
    frame = bars(180)
    baseline = compute(frame)
    for index in (0, 2, 3, 10, 11, 50, 113, 138):
        mutated = frame.copy()
        mutated.loc[index+1:, ["open", "high", "low", "close"]] = [500, 1000, 1, 2]
        assert compute(mutated)[:index+1] == baseline[:index+1]
        assert compute(frame.iloc[:index+1]) == baseline[:index+1]


def test_determinism_byte_equivalent_and_input_unchanged():
    frame = bars()
    original = frame.copy(deep=True)
    a, b = compute(frame), compute(frame)
    assert a == b
    assert [r.model_dump_json().encode() for r in a] == [r.model_dump_json().encode() for r in b]
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(Exception):
        a[0].candle_range = 0


def test_warmup_and_zero_range_are_explicit():
    rows = compute(bars())
    assert rows[0].atr_14 is None
    assert rows[13].atr_14 is None and rows[14].atr_14 is not None
    assert rows[11].prior_high_12 is None and rows[12].prior_high_12 is not None
    assert rows[98].range_percentile_100 is None and rows[99].range_percentile_100 is not None
    assert rows[112].atr_percentile_100 is None and rows[113].feature_ready
    assert not any(r.feature_ready for r in rows[:113])
    frame = bars(160)
    frame[["open", "high", "low", "close"]] = 100
    row = compute(frame)[-1]
    assert row.body_ratio is None and row.close_location is None
    assert row.atr_14 is None and row.displacement_4_atr is None and not row.feature_ready


def test_context_missing_bar_and_partial_start_not_fabricated():
    frame = bars(24)
    rows = compute(frame.drop(index=2))
    assert rows[1].completed_h1_direction is None
    assert rows[9].completed_h3_direction is None
    assert rows[-1].completed_h3_direction == 1
    partial = compute(frame.iloc[1:12])
    assert all(r.completed_h3_direction is None for r in partial)


def test_as_of_excludes_forming_bar_and_matches_completed_prefix():
    frame = bars()
    cutoff = frame.timestamp_utc.iloc[50] + timedelta(minutes=7)
    assert compute(frame, as_of=cutoff) == compute(frame.iloc[:50])


@pytest.mark.parametrize("kind", ["reverse", "duplicate", "naive", "offgrid", "nan", "infinite", "bad_ohlc", "negative"])
def test_invalid_input_rejected(kind):
    frame = bars(20)
    if kind == "reverse": frame = frame.iloc[::-1]
    if kind == "duplicate": frame.loc[1, "timestamp_utc"] = frame.timestamp_utc.iloc[0]
    if kind == "naive": frame.timestamp_utc = frame.timestamp_utc.dt.tz_localize(None)
    if kind == "offgrid": frame.timestamp_utc += timedelta(minutes=1)
    if kind == "nan": frame.loc[1, "high"] = float("nan")
    if kind == "infinite": frame.loc[1, "high"] = float("inf")
    if kind == "bad_ohlc": frame.loc[1, "high"] = 1
    if kind == "negative": frame.loc[1, "low"] = -1
    with pytest.raises(ValueError): compute(frame)


def test_registry_covers_exact_features_and_configuration():
    engine = UniversalFeatureEngine()
    names = {d.name for d in engine.definitions}
    assert len(names) == 28
    assert names == set(FeatureRow.model_fields)-{"timestamp", "symbol", "decision_timeframe", "feature_ready"}
    assert all(d.lookback > 0 and d.description and d.version == 1 for d in engine.definitions)
    assert engine.analysis_timezone == FeatureConfiguration().analysis_timezone
    row = compute(bars(1))[0]
    assert row.timestamp == pd.Timestamp("2024-01-01 00:15Z")
    assert row.analysis_hour == 19


def test_dst_analysis_hour_uses_decision_close():
    frame = bars(2, start="2024-03-10 06:45")
    assert [r.analysis_hour for r in compute(frame)] == [3, 3]


def test_generic_timeframe_symbol_candle_models_and_empty():
    engine = UniversalFeatureEngine("M5", "UTC")
    frame = bars(150, freq="5min")
    models = [Candle(**r, tick_volume=0, spread=0, real_volume=0) for r in frame.to_dict("records")]
    rows = engine.compute(models, symbol="GBPUSD")
    assert rows[-1].symbol == "GBPUSD" and rows[-1].decision_timeframe == "M5"
    assert rows[-1].feature_ready
    assert engine.compute([], symbol="GBPUSD") == ()
    with pytest.raises(ValueError): UniversalFeatureEngine("M0")
    with pytest.raises(ValueError): UniversalFeatureEngine("H3")
    with pytest.raises(ValueError): compute(frame, as_of=pd.Timestamp("2024-01-01"))

def test_momentum_values_independently():
    rows = compute(bars(30))
    frame = bars(30)
    row = rows[-1]
    assert row.displacement_4_atr == (frame.close.iloc[-1]-frame.close.iloc[-5])/row.atr_14
    assert row.displacement_8_atr == (frame.close.iloc[-1]-frame.close.iloc[-9])/row.atr_14


def test_h3_utc_anchor_does_not_follow_analysis_timezone():
    frame = bars(12, start="2024-01-01 01:00")
    assert all(r.completed_h3_direction is None for r in compute(frame))
    # The fragment 01:00..03:00 cannot form the 00:00 UTC bucket.
    full = bars(24, start="2024-01-01 00:00")
    utc = UniversalFeatureEngine(analysis_timezone="UTC").compute(full, symbol="EURUSD")
    ny = compute(full)
    assert [r.completed_h3_direction for r in utc] == [r.completed_h3_direction for r in ny]


def test_selected_m15_market_state_features_are_integrated():
    frame = bars(40)
    row = compute(frame)[-1]
    assert row.er_4 == pytest.approx(1.0)
    assert row.er_16 == pytest.approx(1.0)
    assert row.rv_16 > 0
    assert row.atr_change_1 == pytest.approx(0.0)
    assert row.atr_change_8 == pytest.approx(0.0)
    assert row.slope_atr_4 > 0
    assert row.slope_atr_16 > 0
    assert row.range_pos_16 > 0.5


def test_market_state_feature_warmups_are_explicit():
    rows = compute(bars(30))
    assert rows[3].er_4 is None and rows[4].er_4 is not None
    assert rows[15].er_16 is None and rows[16].er_16 is not None
    assert rows[15].rv_16 is None and rows[16].rv_16 is not None
    assert rows[14].atr_change_1 is None and rows[15].atr_change_1 is not None
    assert rows[21].atr_change_8 is None and rows[22].atr_change_8 is not None
    assert rows[15].range_pos_16 is None and rows[16].range_pos_16 is not None
