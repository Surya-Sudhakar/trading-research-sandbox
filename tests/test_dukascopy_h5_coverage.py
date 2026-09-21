from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sandbox.market_data.dukascopy_csv import aggregate_csv
from sandbox.market_data.gaps import inventory, SessionProfile, ContinuityPolicy
from sandbox.market_state.discovery import dukascopy_h5_coverage as d
from sandbox.market_state.discovery.validation import no_fitting, load_frozen_inputs


def write(path, times, bids=None, asks=None):
    bids = list(bids) if bids is not None else [1.1]*len(times)
    asks = list(asks) if asks is not None else [b+.0001 for b in bids]
    pd.DataFrame({'timestamp':[pd.Timestamp(t).value//1_000_000 for t in times],
                  'askPrice':asks, 'bidPrice':bids}).to_csv(path,index=False)
    return path


def test_epoch_bid_ohlc_order_duplicates_chunk_invariance(tmp_path):
    times=pd.to_datetime(['2016-01-04 00:00Z','2016-01-04 00:00Z','2016-01-04 00:01Z',
                          '2016-01-04 00:15Z','2016-01-04 00:45Z'])
    p=write(tmp_path/'x.csv',times,[1.1,1.3,1.2,1.4,1.5])
    a,ma=aggregate_csv(p,chunksize=2);b,mb=aggregate_csv(p,chunksize=100)
    pd.testing.assert_frame_equal(a,b);assert ma==mb
    assert ma['2016-01']['duplicate_timestamps']==1
    assert a.iloc[0][['open','high','low','close','tick_count']].tolist()==[1.1,1.3,1.1,1.2,3]
    assert len(a)==3 and a.timestamp_utc.iloc[0]==times[0]
    assert a.partial_boundary.tolist()==[False,False,True]


@pytest.mark.parametrize('kind',['backward','crossed','nan','zero','holdout'])
def test_invalid_sources_fail_closed(tmp_path,kind):
    times=pd.date_range('2016-01-04',periods=3,freq='15min',tz='UTC')
    bids=[1.1]*3;asks=[1.2]*3
    if kind=='backward':times=times[::-1]
    if kind=='crossed':asks[1]=1.0
    if kind=='nan':bids[1]=np.nan
    if kind=='zero':bids[1]=0
    if kind=='holdout':times=pd.date_range('2025-01-01',periods=3,freq='15min',tz='UTC')
    with pytest.raises(ValueError):aggregate_csv(write(tmp_path/'x.csv',times,bids,asks),chunksize=2)


def test_partial_boundaries_explicit(tmp_path):
    p=write(tmp_path/'x.csv',pd.to_datetime(['2016-01-04 00:00:00.100Z','2016-01-04 00:15:00Z','2016-01-04 00:30:00Z'],format='mixed'))
    f,_=aggregate_csv(p)
    assert f.partial_boundary.tolist()==[True,False,True]


def test_month_indexing_matches_timestamp_months_across_chunks(tmp_path):
    times=pd.date_range('2016-01-31 23:00',periods=200,freq='15min',tz='UTC')
    p=write(tmp_path/'x.csv',times)
    small,counts=aggregate_csv(p,chunksize=3)
    large,again=aggregate_csv(p,chunksize=1000)
    pd.testing.assert_frame_equal(small,large);assert counts==again
    for month in times.strftime('%Y-%m').unique():
        selected=times[times.strftime('%Y-%m')==month]
        assert counts[month]['raw_tick_count']==len(selected)
        assert counts[month]['first_timestamp']==selected[0].isoformat()
        assert counts[month]['last_timestamp']==selected[-1].isoformat()


def test_frozen_weekend_semantics_verified_vs_unknown():
    f=pd.DataFrame({'timestamp_utc':pd.to_datetime(['2016-01-08 21:45Z','2016-01-10 22:00Z'])})
    meta=SimpleNamespace(timeframe='M15',provider='DUKASCOPY',symbol='EURUSD')
    policy=ContinuityPolicy(reset_missing_minutes=60,version='CONTINUITY_V1')
    known=SessionProfile('fixture','America/New_York',(4,17,0),(6,17,0),'synthetic test only')
    gaps,ids=inventory(f,meta,known,policy)
    assert ids==[0,0] and gaps.classification.iloc[0]=='EXPECTED_WEEKEND'
    gaps,ids=inventory(f,meta,SessionProfile(),policy)
    assert ids==[0,1] and gaps.classification.iloc[0]=='WEEKEND_ASSOCIATED_UNKNOWN'


def test_short_unexplained_gaps_quarantined_not_bridged(tmp_path):
    times=pd.date_range('2016-01-04',periods=300,freq='15min',tz='UTC').delete(100)
    p=write(tmp_path/'x.csv',times,1.1+np.sin(np.arange(len(times))/8)*.001)
    class Never:
        def predict(self,*a,**k):raise AssertionError('unsafe segment inferred')
        def predict_proba(self,*a,**k):raise AssertionError('unsafe segment inferred')
    result=d.source_coverage(p,Never(),SessionProfile(),ContinuityPolicy(reset_missing_minutes=60,version='CONTINUITY_V1'))
    assert result['complete_vectors']==0 and result['quarantined_continuity_segments']==[0]
    assert result['months']['2016-01']['missing_m15_intervals']==1


def test_actual_frozen_inference_zero_fit_and_feature_order(tmp_path):
    times=pd.date_range('2016-01-04',periods=270,freq='15min',tz='UTC')
    p=write(tmp_path/'x.csv',times,1.1+np.sin(np.arange(len(times))/8)*.001)
    with no_fitting() as guard:
        contract,model,_=load_frozen_inputs()
        profile,policy,evidence=d.frozen_continuity(contract)
        result=d.source_coverage(p,model,profile,policy)
    assert guard['fit_calls']==0 and result['complete_vectors']>0
    assert result['feature_names']==contract['feature_schema']['ordered_names']
    assert sum(result['months']['2016-01']['state_counts'].values())==result['complete_vectors']
    assert evidence['policy']['reset_missing_minutes']==60


def test_discovery_does_not_open_2025_or_2026(tmp_path):
    allowed=tmp_path/'eurusd-tick-2024-01-01-2025-01-01.csv';allowed.touch()
    (tmp_path/'eurusd-tick-2025-01-01-2026-01-01.csv').touch()
    (tmp_path/'eurusd-tick-2026-07-27-2026-08-08.csv').touch()
    assert d.discover([tmp_path])==[allowed]


def test_publication_preserves_existing_artifacts(tmp_path):
    p=tmp_path/'report.json';d.publish(p,{'a':1});d.publish(p,{'a':1})
    with pytest.raises(ValueError):d.publish(p,{'a':2})
