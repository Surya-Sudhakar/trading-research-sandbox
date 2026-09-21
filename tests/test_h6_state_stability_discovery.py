import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sandbox.market_state.discovery import h6_state_stability_discovery as h6


def fixture():
    rows = []
    for state, count in h6.STATES.items():
        for i in range(count):
            timestamp = f'{2016+i%7}-01-05T12:00:00+00:00'
            rows.append(dict(entry_time=timestamp, required_state_timestamp=timestamp, state_timestamp=timestamp,
                state_match_status='MATCHED', state_at_entry=state, outcome='WIN' if i%3 else 'LOSS',
                state_age=i+1, bars_since_transition=None if i%4 == 0 else i,
                max_posterior=.8, posterior_margin=.5, posterior_entropy=.2))
    rows += [dict(entry_time='2016-01-04T12:00:00+00:00', state_match_status='NO_COMPLETE_VECTOR_AT_BOUNDARY',
                  state_age='must not consume') for _ in range(1778)]
    return pd.DataFrame(rows)


def test_population_available_cases_determinism_and_no_mutation():
    frame = fixture()
    original = frame.copy(deep=True)
    table, report = h6.analyze(frame)
    again, repeated = h6.analyze(frame)
    assert_frame_equal(frame, original)
    assert_frame_equal(table, again)
    assert report == repeated
    assert report['reconciliation'] == {'total_ledger':4162, 'matched':2384, 'excluded_unmatched':1778, 'state_counts':h6.STATES}
    assert len(table) == 3*5*8
    for row in report['comparisons']:
        for outcome in ('WIN', 'LOSS'):
            assert row[outcome+'_N']+row[outcome+'_missing'] == report['state_outcomes'][row['state']][outcome]
    assert any(r['WIN_missing'] for r in report['comparisons'] if r['variable'] == 'bars_since_transition')


def test_effect_direction_ties_and_small_samples():
    assert h6.effects(pd.Series([1,2]), pd.Series([2,3]))['cliffs_delta'] == .75
    effects = h6.effects(pd.Series([1,2,3]), pd.Series([2,3,4]))
    assert effects['hedges_g'] == pytest.approx(.8)
    assert effects['mean_difference'] == effects['median_difference'] == 1
    assert h6.effects(pd.Series([1,2]), pd.Series([3]))['hedges_g'] is None
    assert h6.effects(pd.Series([1]), pd.Series(dtype=float))['cliffs_delta'] is None
    assert h6.distribution(pd.Series([1,2,3,4,np.nan]))['IQR'] == 1.5


def test_future_columns_never_consumed_and_reference_quantiles_outcome_blind():
    frame = fixture()
    before, summary = h6.analyze(frame)
    frame['future_return'] = object()
    frame['exit_time'] = '2099-01-01'
    frame['profit'] = 'poison'
    after, changed = h6.analyze(frame)
    assert_frame_equal(before, after)
    assert summary == changed
    frame.loc[frame.state_match_status.eq('MATCHED'), 'outcome'] = 'WIN'
    _, relabeled = h6.analyze(frame)
    assert summary['pooled_matched_reference_quantiles'] == relabeled['pooled_matched_reference_quantiles']
    assert relabeled['operations']['threshold_searches'] == 0
    assert 'cutoffs' not in summary and 'filters' not in summary


@pytest.mark.parametrize('change', ['2023', '2015', 'population', 'unmatched_count', 'future_state', 'state_identity'])
def test_invalid_population_or_causality_rejected(change):
    frame = fixture()
    if change in ('2023', '2015'): frame.loc[0, 'entry_time'] = f'{change}-01-01T12:00:00+00:00'
    if change == 'population': frame = frame.iloc[1:]
    if change == 'unmatched_count': frame = frame.iloc[:-1]
    if change == 'future_state': frame.loc[0, 'state_timestamp'] = '2016-01-05T12:15:00+00:00'
    if change == 'state_identity': frame.loc[0, 'state_at_entry'] = 'S1'
    with pytest.raises(ValueError): h6.analyze(frame)


def test_real_counts_no_fitting_and_source_artifacts_unchanged(monkeypatch):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    def forbidden(*args, **kwargs): raise AssertionError('model calls forbidden')
    for cls, names in ((GaussianMixture, ('fit', 'fit_predict', 'predict', 'predict_proba')),
                       (StandardScaler, ('fit', 'fit_transform', 'partial_fit', 'transform'))):
        for name in names: monkeypatch.setattr(cls, name, forbidden)
    paths = set(h6.OUTPUT.glob('h5*')) | set(h6.OUTPUT.glob('state_transition_discovery_v1*'))
    before = {p: h6.sha256_file(p) for p in paths if p.is_file()}
    _, report = h6.analyze(pd.read_csv(h6.LEDGER, usecols=list(h6.INPUT_COLUMNS)))
    assert report['reconciliation']['matched'] == 2384
    assert report['reconciliation']['excluded_unmatched'] == 1778
    assert report['state_outcomes']['S1'] == {'WIN':785, 'LOSS':9}
    assert all(h6.sha256_file(p) == digest for p, digest in before.items())


def test_runner_preserves_sources_and_refuses_overwrite(tmp_path, monkeypatch):
    ledger = tmp_path/'h5_ledger.csv'
    old_report = tmp_path/'h5_summary.json'
    h4 = tmp_path/'state_transition_discovery_v1.json'
    h4.write_text('frozen H4')
    frame = fixture()
    frame.to_csv(ledger, index=False)
    _, expected = h6.analyze(frame)
    old_report.write_text(json.dumps({'provenance': {'trades_csv_sha256':h6.sha256_file(ledger)},
        'by_state': {s:{'wins':v['WIN'], 'losses':v['LOSS']} for s,v in expected['state_outcomes'].items()}}))
    monkeypatch.setattr(h6, 'OUTPUT', tmp_path)
    monkeypatch.setattr(h6, 'LEDGER', ledger)
    monkeypatch.setattr(h6, 'H5_SUMMARY', old_report)
    before = {p:p.read_bytes() for p in tmp_path.iterdir()}
    result = h6.run()
    assert result['reconciliation']['matched'] == 2384
    assert all(p.read_bytes() == value for p,value in before.items())
    with pytest.raises(FileExistsError): h6.run()
