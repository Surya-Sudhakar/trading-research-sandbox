from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sandbox.market_state.discovery.h5_warmup_diagnostic import trace_entry
from sandbox.market_state.universal.engine import UniversalFeatureEngine
from sandbox.market_state.universal.continuity import QualityContext
from sandbox.market_state.universal.volatility import atr_series
from sandbox.market_state.universal.market_metrics import atr_change


def source(n=240, start='2016-01-04 00:00', segment=0):
    c = 1.1+np.sin(np.arange(n)/8)*.001
    return pd.DataFrame({'timestamp_utc': pd.date_range(start, periods=n, freq='15min', tz='UTC'),
                         'open': c, 'high': c+.0002, 'low': c-.0002, 'close': c,
                         'continuity_segment_id': segment})


@pytest.mark.parametrize('count', [0, 1, 14, 15, 18, 19, 20])
def test_exact_atr14_lag4_minimum(count):
    atr = atr_series([1.2]*count, [1.0]*count, [1.1]*count)
    if count:
        value = atr_change(atr, count-1, 4)
        assert (value is not None) == (count >= 19)
    if count >= 15:
        assert atr[13] is None and atr[14] == pytest.approx(.2)


def test_first_defined_value_needs_228_completed_m15_bars():
    f = source()
    before = trace_entry(f, f.timestamp_utc.iloc[226]+timedelta(minutes=15))
    after = trace_entry(f, f.timestamp_utc.iloc[227]+timedelta(minutes=15))
    assert before['completed_h3_bars'] == 18
    assert before['atr_current'] is not None and before['atr_four_h3_bars_ago'] is None
    assert before['reason'] == 'LAGGED_ATR_NOT_SEEDED'
    assert after['completed_h3_bars'] == 19 and after['calculated_h3_atr_change_4'] is not None
    rows = UniversalFeatureEngine().compute(f, symbol='EURUSD')
    assert rows[226].h3_atr_change_4 is None
    assert rows[227].h3_atr_change_4 == after['calculated_h3_atr_change_4']


def test_partial_start_bucket_does_not_count():
    f = source(239, start='2016-01-04 00:15')
    result = trace_entry(f, pd.Timestamp('2016-01-06 09:00', tz='UTC'))
    assert result['completed_h3_bars'] == 18
    assert result['first_possible_ready_time_if_segment_continues'] == '2016-01-06T12:00:00+00:00'


def test_weekend_segment_reset_does_not_reuse_prior_atr():
    left = source()
    right = source(36, start='2016-01-11 00:00', segment=1)
    f = pd.concat([left, right], ignore_index=True)
    trace = trace_entry(f, right.timestamp_utc.iloc[-1]+timedelta(minutes=15))
    assert trace['completed_h3_bars'] == 3 and trace['reason'] == 'ATR_CURRENT_AND_LAG_NOT_SEEDED'
    rows = UniversalFeatureEngine().compute(f, symbol='EURUSD', quality_context=QualityContext())
    assert rows[len(left)-1].h3_atr_change_4 is not None
    assert rows[-1].h3_atr_change_4 is None


def test_future_observations_cannot_change_trace_or_inputs():
    f = source()
    cutoff = pd.Timestamp('2016-01-05 12:07', tz='UTC')
    a = trace_entry(f, cutoff)
    changed = f.copy(deep=True)
    future = changed.timestamp_utc+timedelta(minutes=15) > cutoff.floor('15min')
    changed.loc[future, ['open', 'high', 'low', 'close']] *= 2
    assert a == trace_entry(changed, cutoff)
    assert all(pd.Timestamp(x['close_time']) <= cutoff for x in a['h3_inputs'])
    assert a == trace_entry(f.loc[~future], cutoff)


def test_unrecorded_missing_bar_is_not_silently_bridged():
    f = source().drop(index=50)
    with pytest.raises(ValueError, match='unrecorded gap'):
        trace_entry(f, pd.Timestamp('2016-01-06 09:00', tz='UTC'))


def test_boundary_not_replaced_by_stale_bar():
    f = source().iloc[:100]
    with pytest.raises(ValueError, match='boundary missing'):
        trace_entry(f, pd.Timestamp('2016-01-06 09:00', tz='UTC'))


def test_holdout_and_naive_dates_rejected():
    for entry in ['2025-01-01T00:00:00Z', '2016-01-06 09:00:00']:
        with pytest.raises(ValueError, match='January'):
            trace_entry(source(), entry)
