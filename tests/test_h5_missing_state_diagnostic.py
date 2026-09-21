"""Synthetic diagnostic tests; no feature recomputation, model inference or fitting."""
import copy
from dataclasses import asdict
from datetime import timedelta
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sandbox.market_state.discovery import h5_missing_state_diagnostic as d
from sandbox.market_state.universal.schema import definitions


def fixture_data(*, offset_hours=36, gap=True):
    opened = pd.Timestamp('2016-01-10', tz='UTC')
    times = pd.date_range(opened + d.h5.STEP, periods=400, freq='15min')
    f = pd.DataFrame({'timestamp': times, 'continuity_segment_id': 1,
                      'provider': 'DUKASCOPY', 'symbol': 'EURUSD', 'decision_timeframe': 'M15'})
    for name in d.h5.FEATURES:
        f[name] = 1.0
    required = opened + timedelta(hours=offset_hours)
    f.loc[f.timestamp.eq(required), 'h3_atr_change_4'] = np.nan
    matched = times[-1]
    t = pd.DataFrame({'entry_time': [required.isoformat()]*1778 + [matched.isoformat()]*2384,
                      'required_state_timestamp': [required.isoformat()]*1778 + [matched.isoformat()]*2384,
                      'state_at_entry': [None]*1778 + ['S1']*2384,
                      'state_match_status': [d.UNMATCHED]*1778 + ['MATCHED']*2384})
    meta = {'quality_mode': 'SEGMENT_LOCAL_V1', 'source_timeframe': 'M15',
            'definitions': [asdict(x) for x in definitions('M15')],
            'decision_validation': {'earliest': (opened-timedelta(days=7) if gap else opened).isoformat(),
                'gaps': [{'previous': '2016-01-08T21:45:00+00:00', 'current': opened.isoformat(),
                          'classification': 'UNEXPLAINED_DATA_GAP'}] if gap else []}}
    return t, f, meta


def test_determinism_no_mutation_no_imputation_and_exact_population():
    t, f, m = fixture_data()
    original_t, original_f, original_m = t.copy(deep=True), f.copy(deep=True), copy.deepcopy(m)
    a, summary = d.diagnose(t, f, m)
    b, repeated = d.diagnose(t, f, m)
    assert_frame_equal(a, b)
    assert summary == repeated
    assert_frame_equal(t, original_t)
    assert_frame_equal(f, original_f)
    assert m == original_m
    assert len(a) == summary['unmatched_trades'] == 1778
    assert summary['matched_trades'] == 2384
    assert a.diagnostic_reason.eq('WEEKLY_SEGMENT_WARMUP').all()
    assert a.completed_h3_bars.eq(12).all()
    assert a.bars_since_segment_start.eq(143).all()
    assert a.missing_features.eq('["h3_atr_change_4"]').all()
    assert 'state_at_entry' not in a  # The diagnostic cannot assign replacement states.


@pytest.mark.parametrize('offset, gap, expected', [(36, False, 'INITIAL_SEGMENT_WARMUP'),
                                                 (56.75, True, 'WEEKLY_SEGMENT_WARMUP'),
                                                 (57, True, 'SPECIFIC_FEATURE_INCOMPLETE'),
                                                 (60, True, 'SPECIFIC_FEATURE_INCOMPLETE')])
def test_warmup_vs_unexplained_null(offset, gap, expected):
    a, summary = d.diagnose(*fixture_data(offset_hours=offset, gap=gap))
    assert a.diagnostic_reason.eq(expected).all()
    assert summary['explained_count'] == (0 if expected == 'SPECIFIC_FEATURE_INCOMPLETE' else 1778)


def test_midweek_reset():
    t, f, m = fixture_data()
    shift = timedelta(days=3)
    for name in ('entry_time', 'required_state_timestamp'):
        t[name] = pd.to_datetime(t[name]) + shift
    f.timestamp += shift
    for name in ('previous', 'current'):
        m['decision_validation']['gaps'][0][name] = (pd.Timestamp(m['decision_validation']['gaps'][0][name])+shift).isoformat()
    a, summary = d.diagnose(t, f, m)
    assert a.diagnostic_reason.eq('MIDWEEK_SEGMENT_WARMUP').all()
    assert summary['wednesday_friday_unexplained'] == 0


def test_no_documented_start_is_unknown():
    t, f, m = fixture_data()
    m['decision_validation']['gaps'] = []
    a, summary = d.diagnose(t, f, m)
    assert a.diagnostic_reason.eq('UNKNOWN').all()
    assert a.bars_since_segment_start.isna().all()
    assert summary['unknown_count'] == 1778


@pytest.mark.parametrize('document_gap', [True, False])
def test_missing_boundary_requires_gap_evidence(document_gap):
    t, f, m = fixture_data()
    stamp = pd.Timestamp(t.required_state_timestamp.iloc[0])
    f = f.loc[~f.timestamp.eq(stamp)].copy()
    if document_gap:
        m['decision_validation']['gaps'].append({'previous': (stamp-2*d.h5.STEP).isoformat(),
            'current': stamp.isoformat(), 'classification': 'UNEXPLAINED_DATA_GAP'})
    a, summary = d.diagnose(t, f, m)
    assert a.diagnostic_reason.eq('SOURCE_BAR_GAP' if document_gap else 'NO_FEATURE_ROW_AT_BOUNDARY').all()
    assert a.number_missing_features.isna().all()  # Unknown contents, not fabricated null features.
    assert summary['explained_count'] == (1778 if document_gap else 0)


def test_future_rows_only_change_diagnostic_neighbor_fields():
    t, f, m = fixture_data()
    before, _ = d.diagnose(t, f, m)
    required = pd.Timestamp(t.required_state_timestamp.iloc[0])
    f = f.loc[~f.timestamp.eq(required+d.h5.STEP)].copy()
    after, _ = d.diagnose(t, f, m)
    assert not before.next_available_feature_timestamp.equals(after.next_available_feature_timestamp)
    columns = [c for c in before if c not in ('next_available_feature_timestamp', 'next_complete_vector_timestamp')]
    assert_frame_equal(before[columns], after[columns])


@pytest.mark.parametrize('alter', ['population', 'status', 'complete_vector', 'future_boundary', 'schema'])
def test_mismatch_fails_loudly(alter):
    t, f, m = fixture_data()
    if alter == 'population': t = t.iloc[:-1]
    if alter == 'status': t.loc[0, 'state_match_status'] = 'MATCHED'
    if alter == 'complete_vector': f['h3_atr_change_4'] = 1.0
    if alter == 'future_boundary': t.loc[0, 'required_state_timestamp'] = '2025-01-01T00:00:00+00:00'
    if alter == 'schema': f = f[[c for c in f if c != 'er_4'] + ['er_4']]
    with pytest.raises(ValueError): d.diagnose(t, f, m)


def test_existing_ledger_reconciles_without_changing_matched_results():
    before = {p: d.sha256_file(p) for p in (d.LEDGER, d.SUMMARY)}
    ledger = pd.read_csv(d.LEDGER)
    assert d.reconcile(ledger) == {'total_trades': 4162, 'matched_trades': 2384, 'unmatched_trades': 1778}
    assert {p: d.sha256_file(p) for p in before} == before


def test_runner_create_only_preserves_all_research_inputs(tmp_path, monkeypatch):
    t, f, m = fixture_data()
    ledger, summary = tmp_path/'ledger.csv', tmp_path/'h5_summary.json'
    feature = tmp_path/'features.parquet'
    feature.write_bytes(b'fixture')
    meta = tmp_path/'metadata.json'
    meta.write_text(json.dumps(m))
    model, contract_path, ea = (tmp_path/n for n in ('model.json', 'contract.json', 'ea.mq5'))
    model.write_bytes(b'frozen model')
    ea.write_bytes(b'canonical strategy')
    contract = {'feature_artifact': {'path': str(feature), 'sha256': d.sha256_file(feature),
                                   'metadata_sha256': d.sha256_file(meta)}}
    contract_path.write_text(json.dumps(contract))
    t.to_csv(ledger, index=False)
    summary.write_text(json.dumps({'provenance': {'trades_csv_sha256': d.sha256_file(ledger),
                                                'feature_artifact': contract['feature_artifact']}}))
    monkeypatch.setattr(d, 'OUTPUT', tmp_path)
    monkeypatch.setattr(d, 'LEDGER', ledger)
    monkeypatch.setattr(d, 'SUMMARY', summary)
    monkeypatch.setattr(d, 'validate_contract', lambda _: None)
    monkeypatch.setattr(d, 'EXPECTED_MODEL_SHA256', d.sha256_file(model))
    monkeypatch.setattr(d.h5, 'DEFAULT_MODEL', model)
    monkeypatch.setattr(d.h5, 'DEFAULT_CONTRACT', contract_path)
    monkeypatch.setattr(d.h5, 'EA', ea)
    monkeypatch.setattr(d.h5, 'read_features', lambda *_: f)
    def forbidden(*args, **kwargs): raise AssertionError('state inference forbidden')
    monkeypatch.setattr(d.h5, 'assign_states', forbidden)
    monkeypatch.setattr(d.h5, 'load_frozen_inputs', forbidden)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = d.run()
    assert result['unmatched_trades'] == 1778
    assert result['provenance']['fit_calls'] == result['provenance']['inference_calls'] == 0
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError): d.run()
