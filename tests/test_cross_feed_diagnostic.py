from datetime import timedelta
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from sandbox.market_state.discovery import cross_feed_diagnostic as d
from tests.test_h5_warmup_diagnostic import source


def ic_file(tmp_path, rows):
    p = tmp_path/'ic.csv'
    p.write_text('<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n'+
                 '\n'.join('2026.08.03\t'+r for r in rows))
    return p


def test_utc_and_causal_sparse_quotes(tmp_path):
    p = ic_file(tmp_path, ['12:00:00.100\t\t1.11\t\t\t4',
        '12:00:00.200\t1.10\t\t\t\t2', '12:00:00.200\t\t1.12\t\t\t4',
        '12:00:01.000\t1.09\t\t\t\t2'])
    stats = {}
    values = list(d.ticks(p, 'ic', stats))
    assert values[0][0] == pd.Timestamp('2026-08-03T09:00:00.200Z')
    assert [(x[1], x[2]) for x in values] == [(1.10, 1.11), (1.10, 1.12), (1.09, 1.12)]
    assert stats['bid_unavailable'] == 1 and stats['duplicate_timestamps'] == 1


@pytest.mark.parametrize('row', ['12:00:00.100\t\t1.11\t\t\t2',
    '12:00:00.100\tnan\t1.11\t\t\t6', '12:00:00.100\t-1\t1.11\t\t\t6'])
def test_bad_quotes_rejected(tmp_path, row):
    with pytest.raises(ValueError): list(d.ticks(ic_file(tmp_path, [row]), 'ic', {}))


def duka_file(tmp_path, points):
    path = tmp_path/'duka.csv'
    path.write_text('timestamp,askPrice,bidPrice\n'+'\n'.join(
        f'{pd.Timestamp(t).value//1000000},{bid+.0001},{bid}' for t, bid in points))
    return path


def test_deterministic_ohlc_no_fill_no_partial_bucket(tmp_path):
    p = duka_file(tmp_path, [('2026-08-03T09:00:00Z', 1.1), ('2026-08-03T09:15:00Z', 1.2),
        ('2026-08-03T09:15:00Z', 1.3), ('2026-08-03T09:16:00Z', 1.1),
        ('2026-08-03T09:45:00Z', 1.4), ('2026-08-03T10:00:00Z', 1.5)])
    start = pd.Timestamp('2026-08-03T09:00:01Z'); end = pd.Timestamp('2026-08-03T10:00:00Z')
    f, meta = d.aggregate(p, 'duka', start, end)
    again, second = d.aggregate(p, 'duka', start, end)
    pd.testing.assert_frame_equal(f, again)
    assert meta == second
    assert len(f) == 2 and meta['empty_m15_slots_not_filled'] == 1
    assert f.iloc[0][['open', 'high', 'low', 'close']].tolist() == [1.2, 1.3, 1.1, 1.1]
    assert f.continuity_segment_id.tolist() == [0, 1]


def test_outside_authorized_dates_never_used(tmp_path):
    p = duka_file(tmp_path, [('2025-08-03T09:00:00Z', 1.0), ('2026-08-03T09:00:00Z', 1.1)])
    stats = {}; values = list(d.ticks(p, 'duka', stats))
    assert len(values) == 1 and stats['outside_authorized_window'] == 1


class Model:
    def predict_proba(self, x, *, feature_names):
        assert tuple(feature_names) == d.FEATURES
        return np.tile([.7, .2, .1], (len(x), 1))

    def predict(self, x, *, feature_names):
        return np.zeros(len(x), dtype=int)


def test_engine_reused_missing_vectors_not_imputed_and_prefix_invariant():
    bars = source(260, '2026-08-03 09:15')
    f = d.infer(bars, Model())
    assert f.state.iloc[:227].isna().all()
    assert f.state.notna().any()
    pd.testing.assert_frame_equal(f.iloc[:245].reset_index(drop=True), d.infer(bars.iloc[:245], Model()))
    assert f.loc[f.state.notna(), ['p_S0', 'p_S1', 'p_S2']].iloc[0].tolist() == [.7, .2, .1]


@pytest.mark.parametrize('cls,method', [(StandardScaler, 'fit'), (StandardScaler, 'partial_fit'),
    (GaussianMixture, 'fit'), (GaussianMixture, 'fit_predict')])
def test_training_calls_prohibited(cls, method):
    class Bad(Model):
        def predict_proba(self, x, **kwargs):
            getattr(cls(), method)(x)
    with pytest.raises(RuntimeError, match='fitting'):
        d.infer(source(260, '2026-08-03 09:00'), Bad())


def assigned(times, labels, segments=None):
    n = len(times)
    return pd.DataFrame({'timestamp': pd.to_datetime(times, utc=True),
        'continuity_segment_id': [0]*n if segments is None else segments,
        **{f: np.ones(n) for f in d.FEATURES}, 'unavailable_features': ['']*n, 'state': labels,
        **{p: [.5]*n for p in d.POSTERIORS}})


def test_exact_join_no_nearest_match():
    a = assigned(['2026-08-03 10:00Z', '2026-08-03 10:15Z'], ['S0', 'S1'])
    b = assigned(['2026-08-03 10:00Z', '2026-08-03 10:30Z'], ['S0', 'S2'])
    j = d.join_states(a, b)
    assert len(j) == 3 and j.matched_valid.sum() == 1
    report = d.compare(j)
    assert report['overall_agreement_percent'] == 100
    assert report['unavailable_or_unmatched_union_rows'] == 2


@pytest.mark.parametrize('break_kind', ['gap', 'segment', 'invalid_vector'])
def test_transitions_do_not_bridge_missing_or_disconnected_rows(break_kind):
    times = pd.date_range('2026-08-03 10:00', periods=4, freq='15min', tz='UTC')
    a = assigned(times, ['S0', 'S0', 'S1', 'S1'])
    b = assigned(times, ['S0', 'S0', 'S2', 'S1'])
    if break_kind == 'gap': a = a.drop(index=1)
    if break_kind == 'segment': a['continuity_segment_id'] = [0, 1, 2, 2]
    if break_kind == 'invalid_vector': a.loc[1, 'state'] = None
    report = d.compare(d.join_states(a, b))
    t = report['transitions']
    assert t['consecutive_valid_edges'] == 1
    assert t['duka']['state_changes'] == 0 and t['ic']['state_changes'] == 1


def test_confusion_matrix_and_descriptive_differences():
    times = pd.date_range('2026-08-03 10:00', periods=3, freq='15min', tz='UTC')
    a = assigned(times, ['S0', 'S1', 'S2']); b = assigned(times, ['S0', 'S2', 'S2'])
    b['posterior_entropy'] = [.6, .7, .8]
    j = d.join_states(a, b)
    r = d.compare(j)
    assert r['confusion_matrix_rows_duka_columns_ic'] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert r['overall_agreement_percent'] == pytest.approx(200/3)
    assert r['posterior_absolute_differences']['posterior_entropy']['mean'] == pytest.approx(.2)
    assert r['confidence_comparison_without_cutoff']['duka']['disagree']['max_posterior']['n'] == 1
    json.dumps(r, allow_nan=False)


def test_corrupt_model_sha_rejected(tmp_path):
    p = tmp_path/'model.json'; p.write_text('{}')
    with d.no_fitting():
        with pytest.raises(ValueError, match='SHA256'):
            d.load_frozen_inputs(model_path=p)


def test_contract_tampering_rejected(tmp_path):
    data = json.loads(d.DEFAULT_CONTRACT.read_text())
    data['model_configuration']['n_components'] = 4
    p = tmp_path/'contract.json'; p.write_text(json.dumps(data))
    with d.no_fitting():
        with pytest.raises(ValueError): d.load_frozen_inputs(contract_path=p)


def test_no_inputs_no_output(tmp_path):
    with pytest.raises(FileNotFoundError): d.run(tmp_path/'missing', tmp_path/'other', tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_complete_synthetic_run_uses_frozen_model_and_publishes_outputs(tmp_path):
    times = pd.date_range('2026-08-03 00:00', periods=261, freq='15min', tz='UTC')
    prices = 1.1+np.sin(np.arange(len(times))/8)*.001
    duka = duka_file(tmp_path, list(zip(times, prices)))
    ic = tmp_path/'ic.csv'
    ic.write_text('<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n'+'\n'.join(
        (t+timedelta(hours=3)).strftime('%Y.%m.%d\t%H:%M:%S.%f')+f'\t{bid}\t{bid+.0001}\t\t\t6'
        for t, bid in zip(times, prices)))
    result = d.run(duka, ic, output=tmp_path/'out')
    assert result['fit_calls'] == 0 and result['matched_state_count'] > 0
    assert result['overall_agreement_percent'] == 100
    assert result['model_sha256'] == d.EXPECTED_MODEL_SHA256
    assert result['contract_checksum'] == d.CONTRACT_SHA256
    assert result['protected_before'] == result['protected_after']
    assert result['posterior_absolute_differences']['p_S0']['mean'] == 0
    assert len(pd.read_csv(tmp_path/'out/cross_feed_aug2026_disagreements.csv')) == 0
    for name, digest in result['output_csv_sha256'].items():
        assert d.sha256_file(tmp_path/'out'/name) == digest
