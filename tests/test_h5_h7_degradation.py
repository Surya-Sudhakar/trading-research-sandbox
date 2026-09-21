"""Post-H7 diagnostics never mutate the completed experiments."""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook

from sandbox.market_state.diagnostics import h5_h7_degradation as d


def trades(profits):
    return pd.DataFrame({'profit':profits,'outcome':np.where(np.array(profits)>0,'WIN','LOSS'),
                         'entry_time':pd.date_range('2025-01-01',periods=len(profits),freq='D',tz='UTC')})


def test_exact_symmetric_decomposition_and_input_immutability():
    a,b = trades([1,1,1,-4]),trades([2,2,-5,-5])
    before = a.copy(deep=True)
    result = d.decompose(a,b)
    assert result['change_eur_per_trade'] == pytest.approx(-1.25)
    assert result['contributions'] == pytest.approx({'win_probability':-1.5,'average_winner':.625,'average_loser':-.375})
    assert_frame_equal(a,before)
    reverse = d.decompose(b,a)
    assert reverse['contributions'] == pytest.approx({k:-v for k,v in result['contributions'].items()})


def test_gross_plus_costs_is_alternative_not_double_counted():
    a,b = trades([1,-3]),trades([2,-4])
    for f,c in ((a,-.1),(b,-.2)):
        f['commission_total']=c
        f['swap_total']=0.
        f['gross_profit']=f.profit-c
    gross = d.decompose(a,b,'gross_profit')
    costs = b.commission_total.mean()-a.commission_total.mean()
    assert gross['change_eur_per_trade']+costs == pytest.approx(b.profit.mean()-a.profit.mean())


def test_entry_and_exit_cash_both_included():
    f=trades([.82]);f['entry_deal']=2;f['exit_deal']=3
    result=d.cash_from_deals(f,{2:(0,-.04,0),3:(1,-.04,-.10)})
    assert result.commission_total.iloc[0] == -.08
    assert result.swap_total.iloc[0] == -.10
    assert 'gross_profit' not in f
    with pytest.raises(ValueError,match='reconcile'):
        d.cash_from_deals(f,{2:(0,0,0),3:(1,-.04,-.10)})


def test_report_reader_uses_selected_actual_deals_only(tmp_path):
    f=trades([.82]);f['entry_deal']=2;f['exit_deal']=3
    f['entry_price']=1.1;f['exit_price']=1.1005
    f['exit_time']=f.entry_time+timedelta(minutes=15)
    path=tmp_path/'report.xlsx';w=Workbook();s=w.active
    s.append(['Expert:',None,None,'F1_NY3H_Green_MT5_Version0'])
    s.append(['Currency:',None,None,'EUR'])
    for v in d.h7.INPUTS:s.append([None,None,None,v])
    s.append(['Results']);s.append(['Orders']);s.append(['Deals'])
    s.append(['2025.01.01 00:00:00',2,'EURUSD_DUKASCOPY','buy','in',.01,1.1,1,-.04,0,0])
    s.append(['2025.01.01 00:15:00',3,'EURUSD_DUKASCOPY','sell','out',.01,1.1005,2,-.04,-.1,1])
    # A nonmember row cannot enter the accounting.
    s.append(['2024.01.01 00:00:00',999,'OTHER','sell','out',1,99,9,99,99,99])
    w.save(path);w.close()
    before=d.sha256_file(path)
    result,_=d.report_cash(path,f)
    assert result.gross_profit.tolist()==[1.]
    assert d.sha256_file(path)==before


def test_drift_known_KS_and_no_imputation():
    result = d.drift([0,0,1,np.nan],[1,2,2])
    assert result['KS'] == pytest.approx(2/3)
    assert result['H5']['N']==3 and result['H5']['missing']==1
    assert d.drift([np.nan],[1])['KS'] is None
    assert d.drift([1,1],[1,1])['standardized_mean_difference'] is None
    assert d.drift([0,1],[0,1])['KS']==0


def test_continuity_breaks_runs_and_transitions():
    times = pd.to_datetime(['2025-01-01 00:00Z','2025-01-01 00:15Z','2025-01-01 01:00Z','2025-01-01 01:15Z'])
    f=pd.DataFrame({'state_timestamp':times,'state_at_entry':['S0','S0','S1','S2'],
                    'continuity_segment_id':[0,0,1,2],**{v:[1.]*4 for v in d.VARIABLES}})
    result=d.observation_summary(f)
    assert result['connected_pairs']==1
    assert result['transition_counts']==[[1,0,0],[0,0,0],[0,0,0]]
    assert result['states']['S0']['run_lengths_bars']['mean']==2
    assert sum(v['fraction'] for v in result['states'].values())==1


def test_calendar_retains_all_groups_and_no_cutoffs():
    f=trades([1,-2])
    groups=d.calendar_groups(f)
    assert len(groups['hour_utc'])==24
    assert len(groups['weekday'])==7
    assert len(groups['month'])==12
    assert sum(g['N'] for g in groups['hour_utc'])==2
    assert sum(g['losses'] for g in groups['month'])==1


def test_bootstrap_deterministic():
    f=trades([1,2,-4,1,-2])
    assert d.mean_uncertainty(f)==d.mean_uncertainty(f)


@pytest.mark.parametrize('era', ['H5','H7'])
def test_real_frozen_reconciliation_read_only(era):
    path=d.ROOT/(d.PREFIX[era]+'_trades.csv')
    before=d.sha256_file(path)
    f=d.check_ledger(pd.read_csv(path),era)
    assert len(f)==d.EXPECTED[era][0]
    assert f.state_match_status.eq('MATCHED').sum()==d.EXPECTED[era][1]
    assert d.sha256_file(path)==before
    s1=f.loc[f.state_at_entry.eq('S1')]
    assert (s1.outcome.eq('WIN').sum(),s1.outcome.eq('LOSS').sum())==((785,9) if era=='H5' else (50,13))


@pytest.mark.parametrize('era,bad', [('H5','2023-01-01'),('H7','2024-12-31'),('H7','2026-01-01')])
def test_outside_period_rejected(era,bad):
    f=pd.read_csv(d.ROOT/(d.PREFIX[era]+'_trades.csv'))
    f.loc[0,'entry_time']=bad+' 00:00:00+00:00'
    with pytest.raises(ValueError,match='period'):
        d.check_ledger(f,era)


def test_changed_population_rejected():
    f=pd.read_csv(d.ROOT/(d.PREFIX['H7']+'_trades.csv')).iloc[:-1]
    with pytest.raises(ValueError,match='reconciliation'):
        d.check_ledger(f,'H7')


def test_exact_boundary_never_future_or_stale():
    time=pd.Timestamp('2025-01-01 00:15Z')
    f=pd.DataFrame({'entry_time':[time+timedelta(seconds=30)],'required_state_timestamp':[time],
                    'state_match_status':['MATCHED']})
    features=pd.DataFrame({'timestamp':[time+timedelta(minutes=15)],**{v:[1.] for v in d.h5.FEATURES}})
    with pytest.raises(ValueError,match='exact-boundary'):
        d.entry_features(f,features)


def test_replay_mismatch_fails():
    t=pd.Timestamp('2025-01-01',tz='UTC')
    states=pd.DataFrame({'state_timestamp':[t],'state_at_entry':['S1'],**{v:[.5] for v in ('p_S0','p_S1','p_S2',*d.VARIABLES)}})
    ledger=d.h5.attach(pd.DataFrame({'entry_time':[t]}),states)
    d.verify_state_parity(ledger,states)
    states.loc[0,'state_at_entry']='S2'
    with pytest.raises(ValueError,match='membership'):
        d.verify_state_parity(ledger,states)


@pytest.mark.parametrize('cls',[StandardScaler,GaussianMixture])
def test_fitting_prohibited(cls):
    with d.no_fitting() as guard:
        with pytest.raises(RuntimeError,match='prohibited'):
            cls().fit(np.ones((3,2)))
        assert guard['fit_calls']==1


def test_frozen_loading_never_fits_and_exact_schema():
    with d.no_fitting() as guard:
        contract,model,_=d.load_frozen_inputs()
        assert tuple(contract['feature_schema']['ordered_names'])==d.h5.FEATURES
        assert len(d.h5.FEATURES)==14
    assert guard['fit_calls']==0


def test_outputs_create_only_and_protected_inputs_unchanged(tmp_path):
    source=tmp_path/'h7.json';source.write_text('{"frozen":true}')
    before={str(source):d.sha256_file(source)}
    d.write_outputs({'test':1},[{'N':3}],[{'KS':.2}],tmp_path)
    d.ensure_unchanged(before)
    saved=(tmp_path/d.OUTPUTS[0]).read_bytes()
    with pytest.raises(FileExistsError):
        d.write_outputs({'test':2},[],[],tmp_path)
    assert (tmp_path/d.OUTPUTS[0]).read_bytes()==saved
    source.write_text('{}')
    with pytest.raises(ValueError,match='protected'):
        d.ensure_unchanged(before)
