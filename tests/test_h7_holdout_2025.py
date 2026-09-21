"""H7 pre-execution tests. Actual report checks never calculate holdout states."""
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sandbox.market_state.validation import h7_holdout_2025 as h7
from sandbox.market_state.discovery import h5_discovery_2016_2022 as h5
from sandbox.market_state.discovery.validation import load_frozen_inputs, no_fitting
from sandbox.market_data.gaps import SessionProfile, ContinuityPolicy


class Model:
    def predict_proba(self, x, *, feature_names):
        assert tuple(feature_names) == h7.FEATURES
        labels = np.where(x[:,0] > .6, 1, 2)
        return np.eye(3)[labels]*.8+.2/3

    def predict(self, x, *, feature_names):
        return self.predict_proba(x, feature_names=feature_names).argmax(axis=1)


def bars(start='2025-01-01', n=300):
    close = 1.1+np.arange(n)*.00001+np.sin(np.arange(n)/3)*.0002
    return pd.DataFrame({'timestamp_utc':pd.date_range(start,periods=n,freq='15min',tz='UTC'),
        'open':close, 'high':close+.0001, 'low':close-.0001, 'close':close, 'continuity_segment_id':0})


def report_rows(n=347):
    first = pd.Timestamp('2025-01-03',tz='UTC')
    balance = Decimal('5000')
    deals = [['2025.01.01 00:00:00',1,None,'balance',None,None,None,None,0,0,5000,5000,None,None,None]]
    orders = []
    for i in range(n):
        entry = first+timedelta(hours=i)
        exit = entry+timedelta(minutes=30)
        profit = Decimal('.50') if i%4 else Decimal('-4.00')
        order = 1000+i
        orders.append([entry.strftime('%Y.%m.%d %H:%M:%S'),order,'EURUSD_DUKASCOPY','buy limit',
            '0.01 / 0.01',None,1.1,1.096,1.1005,entry.strftime('%Y.%m.%d %H:%M:%S'),None,'filled','F1 V3',None,None])
        deals.append([entry.strftime('%Y.%m.%d %H:%M:%S'),2*i+2,'EURUSD_DUKASCOPY','buy','in','0.01',1.1,order,
                      0,0,0,float(balance),'F1 V3',None,None])
        balance += profit
        deals.append([exit.strftime('%Y.%m.%d %H:%M:%S'),2*i+3,'EURUSD_DUKASCOPY','sell','out','0.01',
                      1.1005 if profit>0 else 1.096,2000+i,0,0,float(profit),float(balance),'',None,None])
    settings = {'Expert:':'F1_NY3H_Green_MT5_Version0','Symbol:':'EURUSD_DUKASCOPY','Currency:':'EUR',
        'Period:':'M15 (2025.01.01 - 2026.12.31)','Initial Deposit:':5000,'Total Net Profit:':float(balance-Decimal(5000)),
        'Total Trades:':n,'Total Deals:':2*n,'History Quality:':'synthetic fixture','Bars:':300}
    rows = [[name,None,None,value]+[None]*11 for name,value in settings.items()]
    rows += [[None,None,None,value]+[None]*11 for value in h7.INPUTS]
    rows += [['Open Time','Order','Symbol','Type']+[None]*11]+orders
    rows += [['Time','Deal','Symbol','Type','Direction']+[None]*10]+deals
    return rows


def joined_fixture():
    trades,_ = h7.parse_report_rows(report_rows(12))
    trades['required_state_timestamp'] = trades.entry_time.dt.floor('15min')
    trades['state_at_entry'] = [*h7.STATES]*4
    trades['state_match_status'] = 'MATCHED'
    trades.loc[0,'state_at_entry'] = None
    trades.loc[0,'state_match_status'] = 'NO_COMPLETE_VECTOR_AT_BOUNDARY'
    trades['state_diagnostic_reason'] = 'MATCHED'
    trades.loc[0,'state_diagnostic_reason'] = 'SEGMENT_WARMUP'
    trades['missing_features'] = '[]'
    trades.loc[0,'missing_features'] = '["h3_atr_change_4"]'
    return trades


def test_synthetic_694_deals_347_trades_pairing_and_balance():
    trades,summary = h7.parse_report_rows(report_rows())
    assert len(trades) == 347
    assert summary['report_total_trades'] == 347
    assert trades.entry_deal.tolist() == list(range(2,696,2))
    assert trades.exit_deal.tolist() == list(range(3,696,2))
    assert (trades.exit_time-trades.entry_time).eq(timedelta(minutes=30)).all()
    assert summary['report_net_eur'] == round(float(trades.profit.sum()),2)
    assert summary['excluded_outside_2025'] == 0


def test_actual_report_preflight_only_no_state_results():
    before = h7.sha256_file(h7.REPORT)
    trades,summary = h7.read_report(h7.REPORT)
    assert len(trades) == 347 and len(trades)*2 == 694
    assert summary['actual_first_executed_deal'] == '2025-01-03T05:00:30+00:00'
    assert summary['actual_last_executed_deal'] == '2025-12-31T16:00:30+00:00'
    assert trades.entry_time.between(h7.START,h7.END,inclusive='left').all()
    assert before == h7.sha256_file(h7.REPORT)
    assert 'state_at_entry' not in trades


@pytest.mark.parametrize('change',['EA','TP','volume','duplicate','unpaired','balance','total','SL','future_exit'])
def test_invalid_report_rejected(change):
    rows = report_rows(2)
    entry_i = next(i for i,r in enumerate(rows) if r[:5] == ['Time','Deal','Symbol','Type','Direction'])+2
    if change == 'EA': next(r for r in rows if r[0]=='Expert:')[3]='different EA'
    if change == 'TP': next(r for r in rows if r[3]=='TakeProfitPips=5')[3]='TakeProfitPips=6'
    if change == 'volume': rows[entry_i][5]='0.02'
    if change == 'duplicate': rows[entry_i+1][1]=rows[entry_i][1]
    if change == 'unpaired': rows.pop()
    if change == 'balance': rows[entry_i][11]=4999
    if change == 'total': next(r for r in rows if r[0]=='Total Deals:')[3]=999
    if change == 'SL': next(r for r in rows if len(r)>11 and r[3]=='buy limit')[7]=1.095
    if change == 'future_exit': rows[-1][0]='2026.01.01 00:00:00'
    with pytest.raises(ValueError): h7.parse_report_rows(rows)


def test_outside_period_trades_excluded_without_changing_pairing():
    rows = report_rows(2)
    # Move last entire position and associated order outside H7; keep report reconciliation intact.
    for row in rows:
        if len(row)>3 and row[1]==1001 and row[3]=='buy limit':
            row[0]=row[9]='2026.01.01 00:00:00'
    rows[-2][0]='2026.01.01 00:00:00'
    rows[-1][0]='2026.01.01 00:30:00'
    selected,meta=h7.parse_report_rows(rows)
    assert len(selected)==1 and meta['excluded_outside_2025']==1 and meta['report_total_trades']==2


def write_bars(tmp_path, frame):
    f=frame[['open','high','low','close']].copy()
    f.insert(0,'timestamp',frame.timestamp_utc.map(lambda t:t.value//1_000_000))
    path=tmp_path/'input.csv'
    f.to_csv(path,index=False)
    return path


def test_input_schema_continuity_and_immutable_source(tmp_path):
    f=bars(n=20)
    f.loc[10:,'timestamp_utc'] += timedelta(hours=2)
    path=write_bars(tmp_path,f)
    before=path.read_bytes()
    retained,gaps,meta=h7.read_bars(path,SessionProfile(),ContinuityPolicy(reset_missing_minutes=60,version='CONTINUITY_V1'))
    assert retained.continuity_segment_id.tolist()==[0]*10+[1]*10
    assert len(gaps)==1 and meta['M15_rows']==20
    assert path.read_bytes()==before


@pytest.mark.parametrize('change',['schema','2026','2024','invalid_price','short_gap'])
def test_invalid_bar_source_cannot_change_frozen_rules(tmp_path,change):
    f=bars(n=20)
    if change=='2026': f['timestamp_utc'] += timedelta(days=365)
    if change=='2024': f['timestamp_utc'] -= timedelta(days=366)
    if change=='invalid_price': f.loc[0,'high']=0
    if change=='short_gap': f.loc[10:,'timestamp_utc'] += timedelta(minutes=15)
    path=write_bars(tmp_path,f)
    if change=='schema':path.write_text('timestamp,bid\n1735689600000,1.1\n')
    with pytest.raises(ValueError):h7.read_bars(path,SessionProfile(),ContinuityPolicy(reset_missing_minutes=60,version='CONTINUITY_V1'))


def test_frozen_inference_matches_H5_age_transition_and_posteriors():
    source=bars('2016-01-03',n=310)
    original=source.copy(deep=True)
    features,states=h7.state_rows(source,Model())
    comparable=features[['timestamp','continuity_segment_id',*h7.FEATURES]].copy()
    comparable['provider']='DUKASCOPY';comparable['symbol']='EURUSD';comparable['decision_timeframe']='M15'
    expected=h5.assign_states(comparable,Model())
    assert_frame_equal(states[expected.columns],expected)
    assert states.bars_since_transition.iloc[0:1].isna().all()
    assert_frame_equal(source,original)


def test_prefix_features_and_states_are_causal():
    source=bars(n=310)
    all_features,all_states=h7.state_rows(source,Model())
    prefix_features,prefix_states=h7.state_rows(source.iloc[:280],Model())
    assert_frame_equal(all_features.iloc[:280].reset_index(drop=True),prefix_features.reset_index(drop=True))
    last=prefix_features.timestamp.iloc[-1]
    assert_frame_equal(all_states.loc[all_states.state_timestamp<=last].reset_index(drop=True),prefix_states)


def test_exact_boundary_missing_vector_no_stale_or_future_state():
    source=bars(n=310)
    features,states=h7.state_rows(source,Model())
    t=states.state_timestamp.iloc[1]
    # Explicit unavailable vector at t, surrounded by valid states.
    states=states.loc[states.state_timestamp.ne(t)].copy()
    features.loc[features.timestamp.eq(t),'h3_atr_change_4']=np.nan
    trades=pd.DataFrame({'entry_time':[t-timedelta(seconds=1),t,t+timedelta(seconds=1)],
                         'outcome':['WIN']*3,'profit':[.5]*3})
    joined=h7.attach(trades,states,features,source,[asdict(d) for d in h7.definitions('M15')])
    assert joined.state_match_status.tolist()==['MATCHED','NO_COMPLETE_VECTOR_AT_BOUNDARY','NO_COMPLETE_VECTOR_AT_BOUNDARY']
    assert joined.state_timestamp.iloc[0]==t-h7.STEP
    assert joined.state_timestamp.iloc[1:].isna().all()
    assert joined.state_at_entry.iloc[1:].isna().all()


def test_new_segment_warmup_not_carried_forward():
    source=bars(n=330)
    source.loc[300:,'timestamp_utc'] += timedelta(days=2)
    source.loc[300:,'continuity_segment_id']=1
    f,s=h7.state_rows(source,Model())
    assert f.loc[f.continuity_segment_id.eq(1),'state'].isna().all()
    assert not s.continuity_segment_id.eq(1).any()


def test_state_age_unknown_transition_run_preserved(monkeypatch):
    timestamps=pd.date_range('2025-01-03',periods=7,freq='15min',tz='UTC')
    f=pd.DataFrame({'timestamp':timestamps,'continuity_segment_id':[0]*4+[1]*3,
                    'state':['S1','S1','S2','S2','S2','S2','S1']})
    for name in h7.POSTERIORS:f[name]=0.0
    monkeypatch.setattr(h7,'infer',lambda *_:f)
    _,states=h7.state_rows(bars(),Model())
    assert states.state_age.tolist()==[1,2,1,2,1,2,1]
    assert states.bars_since_transition.isna().tolist()==[True,True,False,False,True,True,False]
    assert states.bars_since_transition.dropna().tolist()==[0,1,0]


def test_performance_reconciliation_deterministic_schema_and_no_CI_gate():
    joined=joined_fixture()
    original=joined.copy(deep=True)
    s=h7.summarize(joined)
    assert s==h7.summarize(joined)
    assert_frame_equal(joined,original)
    assert s['reconciliation']=={'total':12,'matched':11,'unmatched':1}
    assert sum(g['N'] for g in s['by_state'].values())==11
    assert len(s['monthly'])==12
    assert s['by_state']['S1']==h5.performance(joined.loc[joined.state_at_entry.eq('S1')])
    expected=all(s['by_state']['S1']['expectancy_eur']>s['by_state'][other]['expectancy_eur'] for other in ('S0','S2'))
    assert s['directional_ordering_reproduced']==expected
    json.dumps(s,allow_nan=False)


def test_frozen_hashes_and_no_fitting_guard(tmp_path):
    with no_fitting() as guard:
        contract,model,_=load_frozen_inputs()
        assert h7.sha256_file(h5.DEFAULT_MODEL)==h7.EXPECTED_MODEL_SHA256
        assert h7.sha256_file(h5.EA)==h5.EA_SHA
        assert contract['contract_sha256']==h7.CONTRACT_SHA256
        x=np.ones((3,14))
        assert model.predict_proba(x,feature_names=h7.FEATURES).shape==(3,3)
    assert guard['fit_calls']==0
    bad=tmp_path/'model.json';bad.write_text('{}')
    with pytest.raises(ValueError,match='SHA256'):load_frozen_inputs(model_path=bad)
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    for cls in (GaussianMixture,StandardScaler):
        with no_fitting(),pytest.raises(RuntimeError,match='fitting is prohibited'):cls().fit(np.ones((3,14)))


def test_one_shot_stops_before_reading_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(h7,'OUTPUT',tmp_path)
    (tmp_path/'h7_holdout_2025.attempt.json').write_text('{"status":"FAILED"}')
    monkeypatch.setattr(h7,'read_report',lambda *_:pytest.fail('report must not be read'))
    with pytest.raises(FileExistsError,match='one-shot'):h7.run()


def test_successful_run_preserves_inputs_and_records_one_shot(tmp_path,monkeypatch):
    output=tmp_path/'output';output.mkdir()
    report,data,ea=(tmp_path/n for n in ('report.xlsx','input.csv','ea.mq5'))
    for p in (report,data,ea):p.write_bytes(b'fixture')
    preserved=output/'h5_frozen.json';preserved.write_text('frozen H5')
    meta=tmp_path/'metadata.json'
    defs=[asdict(d) for d in h7.definitions('M15')]
    meta.write_text(json.dumps({'definitions':defs}))
    trades,_=h7.parse_report_rows(report_rows(12))
    source=bars(n=400)
    joined=joined_fixture()
    monkeypatch.setattr(h7,'OUTPUT',output);monkeypatch.setattr(h7,'REPORT',report);monkeypatch.setattr(h7,'DATA',data)
    monkeypatch.setattr(h5,'EA',ea);monkeypatch.setattr(h5,'EA_SHA',h7.sha256_file(ea))
    monkeypatch.setattr(h7,'load_frozen_inputs',lambda:({'feature_artifact':{'path':str(tmp_path/'features.parquet')}},Model(),'contract-sha'))
    monkeypatch.setattr(h7,'frozen_continuity',lambda _: (SessionProfile(),ContinuityPolicy(),{}))
    monkeypatch.setattr(h7,'read_report',lambda _: (trades,{'reconciled':True}))
    monkeypatch.setattr(h7,'read_bars',lambda *_: (source,pd.DataFrame(),{}))
    monkeypatch.setattr(h7,'state_rows',lambda *_:(pd.DataFrame(),pd.DataFrame()))
    monkeypatch.setattr(h7,'attach',lambda *_:joined)
    before={p:p.read_bytes() for p in (report,data,ea,preserved,meta)}
    result=h7.run()
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert result['reconciliation']=={'total':12,'matched':11,'unmatched':1}
    attempt=json.loads((output/'h7_holdout_2025.attempt.json').read_text())
    assert attempt['status']=='COMPLETED' and attempt['fit_calls']==0
    assert len(attempt['outputs'])==2
    with pytest.raises(FileExistsError):h7.run()
