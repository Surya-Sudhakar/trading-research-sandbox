"""H7-R preparation: report-only real preflight, synthetic state/publishing tests."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sandbox.market_state.validation import h7r_2025 as r
from test_h7_holdout_2025 import Model, bars, report_rows, joined_fixture


def rows_fixture():
    rows=report_rows(4)
    for row in rows:
        for i,value in enumerate(row):
            if value==r.PARSER_SYMBOL:row[i]=r.EXECUTION_SYMBOL
    rows.insert(0,['Profit Trades (% of total):',None,None,'3 (75.00%)']+[None]*11)
    rows.insert(0,['Loss Trades (% of total):',None,None,'1 (25.00%)']+[None]*11)
    rows.insert(0,['Expected Payoff:',None,None,-.625]+[None]*11)
    return rows


def test_adapter_is_specific_and_does_not_mutate_original():
    rows=rows_fixture();before=deepcopy(rows)
    trades,report=r.parse_report_rows(rows)
    assert rows==before
    assert report['execution_symbol']==r.EXECUTION_SYMBOL
    assert (len(trades),report['wins'],report['losses'])==(4,3,1)
    assert report['expected_payoff_eur']==-.625
    with pytest.raises(ValueError,match='noncanonical Symbol'):
        r.h7.parse_report_rows(rows)


@pytest.mark.parametrize('symbol',[r.PARSER_SYMBOL,'EURUSD','ARBITRARY','eurusd_dukascopy_tick'])
def test_other_symbols_rejected(symbol):
    rows=rows_fixture()
    next(row for row in rows if row[0]=='Symbol:')[3]=symbol
    with pytest.raises(ValueError,match='requires exactly'):
        r.parse_report_rows(rows)


@pytest.mark.parametrize('section',['order','deal'])
def test_mixed_symbols_rejected(section):
    rows=rows_fixture()
    row=next(row for row in rows if len(row)>4 and row[2]==r.EXECUTION_SYMBOL
             and (row[3]=='buy limit' if section=='order' else row[3]=='buy'))
    row[2]=r.PARSER_SYMBOL
    with pytest.raises(ValueError,match='symbol mismatch'):
        r.parse_report_rows(rows)


@pytest.mark.parametrize('setting',r.h7.INPUTS)
def test_all_frozen_inputs_still_enforced(setting):
    rows=rows_fixture()
    next(row for row in rows if len(row)>3 and row[3]==setting)[3]=setting+'_changed'
    with pytest.raises(ValueError,match='inputs mismatch'):
        r.parse_report_rows(rows)


@pytest.mark.parametrize('label,value',[('Expert:','OTHER'),('Period:','H1 (2025.01.01 - 2026.01.01)'),
                                       ('Currency:','USD'),('Expected Payoff:',0),
                                       ('Profit Trades (% of total):','4 (100.00%)'),('Total Trades:',99)])
def test_other_report_validation_retained(label,value):
    rows=rows_fixture()
    next(row for row in rows if row[0]==label)[3]=value
    with pytest.raises(ValueError):r.parse_report_rows(rows)


def test_actual_report_reconciliation_only_no_replay(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('real state replay forbidden in preparation')
    monkeypatch.setattr(r.h7,'state_rows',forbidden)
    monkeypatch.setattr(r.h7,'read_bars',forbidden)
    before=r.verify_frozen()
    with r.h7.no_fitting() as guard:
        trades,report=r.read_report()
    assert guard['fit_calls']==0
    assert (len(trades),report['entries'],report['exits'])==(594,594,594)
    assert (report['wins'],report['losses'])==(525,69)
    assert report['net_profit_eur']==-1.82
    assert report['expected_payoff_eur']==pytest.approx(-1.82/594)
    assert report['report_expected_payoff_eur']==-.003064
    assert report['fully_reconciled'] is True
    r.assert_unchanged(before)


@pytest.mark.parametrize('entry',['2025-01-04 10:37:22+00:00','2025-01-04 10:30:00+00:00'])
def test_exact_alignment_and_no_post_entry_information(entry):
    entry=pd.Timestamp(entry);boundary=pd.Timestamp('2025-01-04 10:30:00',tz='UTC')
    source=bars(n=400)
    full_features,full_states=r.h7.state_rows(source,Model())
    prefix=source.loc[source.timestamp_utc+timedelta(minutes=15)<=entry]
    prefix_features,prefix_states=r.h7.state_rows(prefix,Model())
    assert prefix.timestamp_utc.iloc[-1]==pd.Timestamp('2025-01-04 10:15:00',tz='UTC')
    selected=full_features.loc[full_features.timestamp.eq(boundary)].reset_index(drop=True)
    assert_frame_equal(selected,prefix_features.loc[prefix_features.timestamp.eq(boundary)].reset_index(drop=True))
    assert_frame_equal(full_states.loc[full_states.state_timestamp.eq(boundary)].reset_index(drop=True),
                       prefix_states.loc[prefix_states.state_timestamp.eq(boundary)].reset_index(drop=True))
    definitions=[asdict(d) for d in r.h7.definitions('M15')]
    joined=r.h7.attach(pd.DataFrame({'entry_time':[entry]}),full_states,full_features,source,definitions)
    assert joined.state_timestamp.iloc[0]==boundary
    assert joined.state_match_status.iloc[0]=='MATCHED'
    assert joined.state_timestamp.iloc[0]<=entry
    # Change every bar not completed at entry, including the 10:30-open bar.
    altered=source.copy(deep=True)
    altered.loc[altered.timestamp_utc+timedelta(minutes=15)>entry,['open','high','low','close']]+=10
    altered_features,altered_states=r.h7.state_rows(altered,Model())
    assert_frame_equal(selected,altered_features.loc[altered_features.timestamp.eq(boundary)].reset_index(drop=True))
    assert_frame_equal(full_states.loc[full_states.state_timestamp.eq(boundary)].reset_index(drop=True),
                       altered_states.loc[altered_states.state_timestamp.eq(boundary)].reset_index(drop=True))


def test_missing_boundary_never_falls_back():
    time=pd.Timestamp('2025-01-04 10:30:00',tz='UTC')
    states=pd.DataFrame({'state_timestamp':[time-timedelta(minutes=15),time+timedelta(minutes=15)],
                         'state_at_entry':['S1','S2']})
    joined=r.h7.h5.attach(pd.DataFrame({'entry_time':[time+timedelta(seconds=10)]}),states)
    assert joined.state_match_status.iloc[0]=='NO_COMPLETE_VECTOR_AT_BOUNDARY'


def test_frozen_inputs_and_contract_load_without_fitting():
    r.verify_frozen()
    with r.h7.no_fitting() as guard:
        contract,_,_=r.h7.load_frozen_inputs()
    assert guard['fit_calls']==0
    assert tuple(contract['feature_schema']['ordered_names'])==r.h7.FEATURES


def test_checksum_failure_blocks_replay(tmp_path):
    fake=tmp_path/'report.xlsx';fake.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA256 mismatch'):r.verify_frozen(fake)


def test_create_only_stops_before_inputs(tmp_path,monkeypatch):
    path=tmp_path/r.NAMES[2];path.write_text('existing')
    def forbidden(*a,**k):raise AssertionError('must not read inputs')
    monkeypatch.setattr(r,'protected_snapshot',forbidden)
    with pytest.raises(FileExistsError):r.run(output=tmp_path)
    assert path.read_text()=='existing'


@pytest.mark.parametrize('failure',[False,True])
def test_synthetic_orchestration_only_separate_outputs(tmp_path,monkeypatch,failure):
    # All observation/state functions are synthetic stubs. No real replay here.
    old=tmp_path/'h7_holdout_2025_summary.json';old.write_text('frozen')
    source=tmp_path/'features.parquet';source.write_bytes(b'fixture')
    meta=source.with_name('metadata.json')
    meta.write_text(json.dumps({'definitions':[asdict(d) for d in r.h7.definitions('M15')]}))
    frozen={str(old):r.sha256_file(old)}
    monkeypatch.setattr(r,'protected_snapshot',lambda:dict(frozen))
    monkeypatch.setattr(r,'verify_frozen',lambda *a:dict(frozen))
    contract={'feature_artifact':{'path':str(source),'sha256':r.sha256_file(source),'metadata_sha256':r.sha256_file(meta)}}
    monkeypatch.setattr(r.h7,'load_frozen_inputs',lambda:(contract,object(),'contract-hash'))
    monkeypatch.setattr(r.h7,'frozen_continuity',lambda c:(None,None,{}))
    trades,report=r.parse_report_rows(rows_fixture())
    monkeypatch.setattr(r,'read_report',lambda *a:(trades,report))
    monkeypatch.setattr(r.h7,'read_bars',lambda *a:(bars(n=400),None,{}))
    called=[]
    def state_rows(*a):
        called.append('state_rows')
        if failure:raise ValueError('synthetic state failure')
        return pd.DataFrame(),pd.DataFrame()
    monkeypatch.setattr(r.h7,'state_rows',state_rows)
    fixture=joined_fixture()
    monkeypatch.setattr(r.h7,'attach',lambda *a:fixture)
    original_summarize=r.h7.summarize
    def summarize(f):
        called.append('summarize');assert_frame_equal(f,fixture)
        return original_summarize(f)
    monkeypatch.setattr(r.h7,'summarize',summarize)
    if failure:
        with pytest.raises(ValueError,match='synthetic state failure'):r.run(output=tmp_path)
    else:
        summary=r.run(output=tmp_path)
        assert summary['primary_comparisons']==original_summarize(fixture)['primary_comparisons']
        assert summary['provenance']['execution_symbol']==r.EXECUTION_SYMBOL
        assert summary['provenance']['fit_calls']==0
    attempt=json.loads((tmp_path/r.NAMES[2]).read_text())
    assert attempt['execution_symbol']==r.EXECUTION_SYMBOL
    assert attempt['status']==('FAILED' if failure else 'COMPLETED')
    assert (tmp_path/r.NAMES[0]).exists()==(not failure)
    assert (tmp_path/r.NAMES[1]).exists()==(not failure)
    assert old.read_text()=='frozen'
    assert called==(['state_rows'] if failure else ['state_rows','summarize'])
