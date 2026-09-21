from decimal import Decimal
from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from sandbox.market_state.discovery import h5_january as h5


class Model:
    def predict_proba(self, x, *, feature_names):
        assert tuple(feature_names) == h5.FEATURES
        return np.eye(3)[x[:, 0].astype(int)]

    def predict(self, x, *, feature_names):
        return self.predict_proba(x, feature_names=feature_names).argmax(axis=1)


def features(labels=(0, 0, 1, 1, 2)):
    n = len(labels)
    f = pd.DataFrame({'timestamp': pd.date_range('2016-01-05 09:00', periods=n, freq='15min', tz='UTC'),
                      'continuity_segment_id': 0, 'provider': 'DUKASCOPY', 'symbol': 'EURUSD',
                      'decision_timeframe': 'M15', **{c: np.ones(n) for c in h5.FEATURES}})
    f[h5.FEATURES[0]] = np.asarray(labels, dtype=float)
    return f


def report_rows():
    rows = []
    for name, value in {'Expert:': 'F1_NY3H_Green_MT5_Version0', 'Symbol:': 'EURUSD_DUKASCOPY',
                        'Period:': 'M15 (2016.01.01 - 2016.02.01)', 'Currency:': 'EUR',
                        'Initial Deposit:': 5000, 'Total Net Profit:': 1.22, 'Total Trades:': 49,
                        'Total Deals:': 98, 'History Quality:': '93% real ticks'}.items():
        rows.append([name, None, None, value]+[None]*11)
    for value in ('TakeProfitPips=5', 'StopLossPips=40', 'PipSize=0.0001', 'FixedLots=0.01',
                  'MagicNumber=26082930', 'EnableTrading=true', 'UseManualServerUTCOffset=true',
                  'ManualServerUTCOffsetHours=0'):
        rows.append([None, None, None, value]+[None]*11)
    rows.append(['Time', 'Deal', 'Symbol', 'Type', 'Direction']+[None]*10)
    balance = Decimal('5000')
    for i in range(49):
        t = pd.Timestamp('2016-01-05')+timedelta(minutes=i*10)
        entry = [t.strftime('%Y.%m.%d %H:%M:%S'), 2*i+2, 'EURUSD_DUKASCOPY', 'buy', 'in',
                 '.01', 1.1, 2*i+2, 0, 0, 0, float(balance), '', None, None]
        profit = Decimal('.50') if i < 43 else Decimal('-3.38')
        balance += profit
        exit = [(t+timedelta(minutes=1)).strftime('%Y.%m.%d %H:%M:%S'), 2*i+3,
                'EURUSD_DUKASCOPY', 'sell', 'out', '.01', 1.1005, 2*i+3, 0, 0,
                float(profit), float(balance), '', None, None]
        rows.extend([entry, exit])
    return rows


def test_deals_reconcile_and_use_actual_fill():
    rows = report_rows()
    rows.insert(0, ['2016.01.05 00:00:00', 1000, 'EURUSD_DUKASCOPY', 'buy limit']+[None]*11)
    frame, report = h5.parse_report_rows(rows)
    assert len(frame) == 49 and frame.entry_time.iloc[0] == pd.Timestamp('2016-01-05', tz='UTC')
    assert report['net_profit_eur'] == 1.22
    assert report['wins'] == 43 and report['losses'] == 6


@pytest.mark.parametrize('field,value', [('Period:', 'M15 (2025.01.01 - 2025.02.01)'),
    ('Symbol:', 'EURUSD'), ('Total Net Profit:', 2), ('Total Trades:', 48), ('Currency:', 'USD')])
def test_wrong_report_rejected(field, value):
    rows = report_rows()
    next(r for r in rows if r[0] == field)[3] = value
    with pytest.raises(ValueError):
        h5.parse_report_rows(rows)


@pytest.mark.parametrize('change', ['duplicate', 'overlap', 'missing_exit', 'future', 'bad_balance', 'partial'])
def test_bad_deals_rejected(change):
    rows = report_rows()
    start = next(i for i, r in enumerate(rows) if r[0] == 'Time')+1
    if change == 'duplicate': rows[start+1][1] = rows[start][1]
    if change == 'overlap': rows[start+1][3:5] = ['buy', 'in']
    if change == 'missing_exit': rows.pop()
    if change == 'future': rows[start][0] = '2025.01.01 00:00:00'
    if change == 'bad_balance': rows[start][11] = 999
    if change == 'partial': rows[start][5] = '.005'
    with pytest.raises(ValueError): h5.parse_report_rows(rows)


def test_inputs_and_state_identities_unchanged():
    f = features()
    original = f.copy(deep=True)
    s = h5.assign_states(f, Model())
    pd.testing.assert_frame_equal(f, original)
    assert s.state_at_entry.tolist() == ['S0', 'S0', 'S1', 'S1', 'S2']
    assert s.state_age.tolist() == [1, 2, 1, 2, 1]
    assert s.bars_since_transition.iloc[:2].isna().all()
    assert s.bars_since_transition.iloc[2:].tolist() == [0, 1, 0]
    assert s.posterior_entropy.eq(0).all() and s.posterior_margin.eq(1).all()


@pytest.mark.parametrize('kind', ['gap', 'segment', 'missing_vector'])
def test_disconnections_reset_age(kind):
    f = features((0, 1, 1, 1, 1))
    if kind == 'gap': f = f.drop(index=2)
    if kind == 'segment': f.loc[3:, 'continuity_segment_id'] = 1
    if kind == 'missing_vector': f.loc[2, h5.FEATURES[1]] = np.nan
    s = h5.assign_states(f, Model())
    r = s.loc[s.state_timestamp == pd.Timestamp('2016-01-05 09:45', tz='UTC')].iloc[0]
    assert r.state_age == 1 and pd.isna(r.bars_since_transition)


def test_prefix_invariance_and_chronological_sort():
    f = features()
    a = h5.assign_states(f, Model())
    b = h5.assign_states(f.iloc[:3], Model())
    pd.testing.assert_frame_equal(a.iloc[:3].reset_index(drop=True), b)
    pd.testing.assert_frame_equal(a, h5.assign_states(f.iloc[::-1], Model()))


@pytest.mark.parametrize('kind', ['order', 'outcome', 'naive', 'holdout', 'duplicate', 'infinity', 'source'])
def test_feature_guards(kind):
    f = features()
    if kind == 'order': f = f[list(f.columns[:-2])+[f.columns[-1], f.columns[-2]]]
    if kind == 'outcome': f['profit'] = 1
    if kind == 'naive': f.timestamp = f.timestamp.dt.tz_localize(None)
    if kind == 'holdout': f.loc[0, 'timestamp'] = pd.Timestamp('2025-01-01', tz='UTC')
    if kind == 'duplicate': f.loc[1, 'timestamp'] = f.timestamp.iloc[0]
    if kind == 'infinity': f.loc[0, h5.FEATURES[0]] = np.inf
    if kind == 'source': f['provider'] = 'IC_MARKETS'
    with pytest.raises(ValueError): h5.assign_states(f, Model())


def test_entry_boundary_no_future_or_stale_match():
    states = h5.assign_states(features((0, 1, 2)), Model())
    t = pd.DataFrame({'entry_time': pd.to_datetime(['2016-01-05 08:59:59Z', '2016-01-05 09:14:59Z',
                                                  '2016-01-05 09:15:00Z', '2016-01-05 09:45:00Z'])})
    joined = h5.attach(t, states)
    assert joined.state_at_entry.iloc[1:3].tolist() == ['S0', 'S1']
    assert joined.state_at_entry.iloc[[0, 3]].isna().all()


def test_predict_cannot_fit():
    from sklearn.preprocessing import StandardScaler
    class Bad(Model):
        def predict(self, x, **kwargs):
            StandardScaler().fit(x)
    with pytest.raises(RuntimeError, match='fitting'):
        h5.assign_states(features(), Bad())


def test_reader_predicate_and_checksum(monkeypatch, tmp_path):
    path = tmp_path/'f'
    path.write_bytes(b'fixture')
    contract = {'feature_artifact': {'sha256': h5.sha256_file(path)}}
    class Schema:
        names = list(features().columns)
    monkeypatch.setattr(h5.pq, 'read_schema', lambda p: Schema())
    calls = []
    def read(p, **kwargs):
        calls.append(kwargs)
        return features()
    monkeypatch.setattr(h5.pd, 'read_parquet', read)
    h5.read_features(path, contract)
    assert calls[0]['filters'] == [('timestamp', '>=', h5.START), ('timestamp', '<', h5.END)]
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'): h5.read_features(path, contract)
    assert len(calls) == 1


def test_missing_preserved_summary_and_bootstrap_deterministic():
    f = features()
    states = h5.assign_states(f, Model())
    trades = pd.DataFrame({'entry_time': pd.to_datetime(['2016-01-05 09:00Z', '2016-01-05 09:30Z', '2016-01-06 09:00Z']),
                           'profit': [1., -2., 1.], 'outcome': ['WIN', 'LOSS', 'WIN']})
    joined = h5.attach(trades, states)
    c = h5.comparisons(joined)
    assert len(joined) == 3 and c['WIN']['state_distribution']['UNMATCHED'] == 1
    assert c['WIN']['metrics']['posterior_entropy']['missing'] == 1
    a = h5.bootstrap(joined, replicates=40)
    assert a == h5.bootstrap(joined, replicates=40)
    json.dumps(a, allow_nan=False)
    assert a['performance']['S0']['WR']['percentile_95_ci'] is None


def test_refuses_overwriting_existing_output(tmp_path):
    (tmp_path/'h5_january_2016_summary.json').write_text('preserve')
    with pytest.raises(FileExistsError): h5.run(output=tmp_path)
    assert (tmp_path/'h5_january_2016_summary.json').read_text() == 'preserve'


def test_run_publishes_reconciled_outputs_with_post_hash_check(monkeypatch, tmp_path):
    def sha(path):
        assert isinstance(path, Path)
        return h5.EA_SHA if path == h5.EA else 'a'*64
    monkeypatch.setattr(h5, 'sha256_file', sha)
    monkeypatch.setattr(h5, 'OUTPUT', tmp_path)
    monkeypatch.setattr(h5, 'load_frozen_inputs', lambda: (
        {'feature_artifact': {'path': 'fixture', 'sha256': 'a'*64}}, Model(), 'b'*64))
    monkeypatch.setattr(h5, 'read_features', lambda *a: features())
    monkeypatch.setattr(h5, 'read_report', lambda *a: h5.parse_report_rows(report_rows()))
    monkeypatch.setattr(h5, 'bootstrap', lambda f: {'synthetic_test': True})
    result = h5.run(report=Path('fixture.xlsx'), output=tmp_path)
    assert result['overall']['N'] == 49
    assert result['provenance']['protected_before'] == result['provenance']['protected_after']
    assert result['provenance']['fit_calls'] == 0
    assert result['pipeline_status'] == 'FAILED_INCOMPLETE_STATE_COVERAGE'
    assert len(pd.read_csv(tmp_path/'h5_january_2016_trades.csv')) == 49
    assert json.loads((tmp_path/'h5_january_2016_summary.json').read_text()) == result
