import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sandbox.market_state.discovery import h5_temporal_stability as s


def fixture():
    rows = []
    for state, n in s.EXPECTED.items():
        for i in range(n):
            rows.append({'entry_time': f'{2016+i%7}-01-05T12:00:00+00:00',
                         'state_at_entry': state, 'state_match_status': 'MATCHED',
                         'outcome': 'WIN' if i%2 else 'LOSS', 'profit': '1.25' if i%2 else '-2.50'})
    rows.append({'entry_time': '2016-02-01T00:00:00+00:00', 'state_at_entry': None,
                 'state_match_status': 'NO_COMPLETE_VECTOR_AT_BOUNDARY', 'profit': 'not consumed', 'outcome': None})
    return pd.DataFrame(rows)


def test_counts_unmatched_exclusion_determinism_and_no_mutation():
    data = fixture()
    before = data.copy(deep=True)
    table, summary = s.analyze(data)
    again, repeated = s.analyze(data)
    assert_frame_equal(data, before)
    assert_frame_equal(table, again)
    assert summary == repeated
    assert summary['reconciliation']['matched_total'] == 2384
    assert summary['reconciliation']['state_counts'] == s.EXPECTED
    assert summary['reconciliation']['yearly_N_sum'] == 2384
    assert summary['reconciliation']['monthly_S1_N_sum'] == 794
    assert len(table) == 21+84
    assert summary['monthly_S1_stability']['no_trades'] == 77
    assert table.loc[table.N.eq(0), 'expectancy_eur_per_trade'].isna().all()


def test_money_win_rates_and_contributions():
    _, summary = s.analyze(fixture())
    assert summary['overall_by_state']['S1'] == {
        'N': 794, 'wins': 397, 'losses': 397, 'win_rate': .5,
        'net_profit_eur': -496.25, 'expectancy_eur_per_trade': -.625}
    assert sum(r['s1_total_trade_share_percent'] for r in summary['yearly_S1']) == pytest.approx(100)
    assert sum(r['s1_total_net_profit_share_percent'] for r in summary['yearly_S1']) == pytest.approx(100)
    for year in s.YEARS:
        assert sum(r['yearly_matched_trade_share_percent'] for r in summary['yearly_states'] if r['period'] == str(year)) == pytest.approx(100)


@pytest.mark.parametrize('change', ['2023', '2015', 'count', 'state', 'profit_sign'])
def test_rejects_wrong_period_or_population(change):
    data = fixture()
    if change in ('2023', '2015'): data.loc[0, 'entry_time'] = f'{change}-01-01T00:00:00+00:00'
    if change == 'count': data = data.iloc[1:]
    if change == 'state': data.loc[0, 'state_at_entry'] = 'S1'
    if change == 'profit_sign': data.loc[0, 'profit'] = '9.0'
    with pytest.raises(ValueError): s.analyze(data)


def test_actual_h5_reconciliation_and_unchanged_artifacts():
    paths = [s.LEDGER, s.H5_SUMMARY]
    before = {p: s.sha256_file(p) for p in paths}
    _, summary = s.analyze(pd.read_csv(s.LEDGER, dtype={'profit': str}))
    assert summary['reconciliation']['matched_total'] == 2384
    assert summary['reconciliation']['state_counts'] == {'S0':114, 'S1':794, 'S2':1476}
    assert summary['reconciliation']['excluded_unmatched'] == 1778
    assert {p: s.sha256_file(p) for p in paths} == before


def test_runner_create_only_preserves_sources(tmp_path, monkeypatch):
    data = fixture()
    ledger, report = tmp_path/'h5_ledger.csv', tmp_path/'h5_report.json'
    data.to_csv(ledger, index=False)
    _, expected = s.analyze(data)
    report.write_text(json.dumps({'provenance': {'trades_csv_sha256': s.sha256_file(ledger)},
                                 'by_state': expected['overall_by_state']}))
    monkeypatch.setattr(s, 'OUTPUT', tmp_path)
    monkeypatch.setattr(s, 'LEDGER', ledger)
    monkeypatch.setattr(s, 'H5_SUMMARY', report)
    before = {p: p.read_bytes() for p in (ledger, report)}
    result = s.run()
    assert result['reconciliation']['matched_total'] == 2384
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(FileExistsError): s.run()
