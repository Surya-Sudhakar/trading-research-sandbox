"""Read-only, entry-calendar summaries of the existing matched H5 ledger."""
from decimal import Decimal
import json
from pathlib import Path

import pandas as pd

from sandbox.provenance import sha256_file

OUTPUT = Path('results/market_state')
LEDGER = OUTPUT/'h5_discovery_2016_2022_trades.csv'
H5_SUMMARY = OUTPUT/'h5_discovery_2016_2022_summary.json'
EXPECTED = {'S0': 114, 'S1': 794, 'S2': 1476}
YEARS = range(2016, 2023)


def metrics(frame):
    n = len(frame)
    wins = int(frame.outcome.eq('WIN').sum())
    net = sum(frame['_cash'], Decimal(0))
    return {'N': n, 'wins': wins, 'losses': n-wins,
            'win_rate': wins/n if n else None, 'net_profit_eur': float(net),
            'expectancy_eur_per_trade': float(net/n) if n else None}


def analyze(trades):
    """No model, features or strategy execution; unmatched outcomes are not consumed."""
    timestamps = pd.DatetimeIndex(trades.entry_time)
    if timestamps.tz is None or timestamps.hasnans:
        raise ValueError('valid aware entry timestamps required')
    timestamps = timestamps.tz_convert('UTC')
    if not ((timestamps >= pd.Timestamp('2016-01-01', tz='UTC')) &
            (timestamps < pd.Timestamp('2023-01-01', tz='UTC'))).all():
        raise ValueError('only 2016-2022 discovery entries allowed')
    selected = trades.state_match_status.eq('MATCHED')
    matched = trades.loc[selected].copy()
    matched['entry_time'] = timestamps[selected]
    counts = {str(k): int(v) for k, v in matched.state_at_entry.value_counts(dropna=False).items()}
    if len(matched) != 2384 or counts != EXPECTED:
        raise ValueError(f'matched H5 reconciliation failed: {len(matched)}, {counts}')
    matched['_cash'] = matched.profit.map(lambda x: Decimal(str(x)))
    if (not matched['_cash'].map(lambda x: x.is_finite()).all() or
            not matched.outcome.isin(['WIN', 'LOSS']).all()):
        raise ValueError('invalid matched outcome/profit')
    if not all((cash > 0) == (outcome == 'WIN') and cash != 0
               for cash, outcome in zip(matched['_cash'], matched.outcome)):
        raise ValueError('matched outcome/profit sign mismatch')
    matched['_year'] = matched.entry_time.dt.year
    matched['_month'] = matched.entry_time.dt.strftime('%Y-%m')
    yearly = []
    s1 = matched.loc[matched.state_at_entry.eq('S1')]
    s1_net = sum(s1['_cash'], Decimal(0))
    for year in YEARS:
        year_frame = matched.loc[matched['_year'].eq(year)]
        for state in EXPECTED:
            group = year_frame.loc[year_frame.state_at_entry.eq(state)]
            row = {'period_type': 'YEAR', 'period': str(year), 'state': state, **metrics(group),
                   'yearly_matched_trade_share_percent': 100*len(group)/len(year_frame) if len(year_frame) else None,
                   's1_total_trade_share_percent': 100*len(group)/len(s1) if state == 'S1' else None,
                   's1_total_net_profit_share_percent': float(100*sum(group['_cash'], Decimal(0))/s1_net)
                   if state == 'S1' and s1_net else None}
            yearly.append(row)
    monthly = [{'period_type': 'MONTH', 'period': f'{year}-{month:02d}', 'state': 'S1',
                **metrics(s1.loc[s1['_month'].eq(f'{year}-{month:02d}')])}
               for year in YEARS for month in range(1, 13)]
    yearly_s1 = [r for r in yearly if r['state'] == 'S1']
    def signs(rows):
        return {'positive': sum(r['N'] > 0 and r['net_profit_eur'] > 0 for r in rows),
                'negative': sum(r['N'] > 0 and r['net_profit_eur'] < 0 for r in rows),
                'zero': sum(r['N'] > 0 and r['net_profit_eur'] == 0 for r in rows),
                'no_trades': sum(r['N'] == 0 for r in rows)}
    def leave_out(rows):
        return [{'excluded_period': r['period'],
                 'remaining_N': len(s1)-r['N'],
                 'remaining_net_profit_eur': float(s1_net-Decimal(str(r['net_profit_eur']))),
                 'remaining_expectancy_eur_per_trade': float((s1_net-Decimal(str(r['net_profit_eur'])))/(len(s1)-r['N']))
                 if len(s1) != r['N'] else None} for r in rows]
    populated = [r for r in monthly if r['N']]
    summary = {'experiment': 'H5_TEMPORAL_STABILITY_V1',
        'reconciliation': {'matched_total': len(matched), 'state_counts': counts,
                           'excluded_unmatched': int((~selected).sum()),
                           'yearly_N_sum': sum(r['N'] for r in yearly),
                           'monthly_S1_N_sum': sum(r['N'] for r in monthly)},
        'overall_by_state': {state: metrics(matched.loc[matched.state_at_entry.eq(state)]) for state in EXPECTED},
        'yearly_states': yearly, 'yearly_S1': yearly_s1,
        'expectancy_sign_year_counts': {state: signs([r for r in yearly if r['state'] == state]) for state in EXPECTED},
        'monthly_S1': monthly,
        'monthly_S1_stability': {**signs(monthly), 'months_with_trades': len(populated),
            'N_min': min(r['N'] for r in populated), 'N_max': max(r['N'] for r in populated),
            'N_median': float(pd.Series([r['N'] for r in populated]).median()),
            'win_rate_min': min(r['win_rate'] for r in populated),
            'win_rate_max': max(r['win_rate'] for r in populated),
            'expectancy_min': min(r['expectancy_eur_per_trade'] for r in populated),
            'expectancy_max': max(r['expectancy_eur_per_trade'] for r in populated)},
        'S1_concentration': {
            'largest_trade_count_year': max(yearly_s1, key=lambda r: r['N']),
            'largest_net_profit_year': max(yearly_s1, key=lambda r: r['net_profit_eur']),
            'largest_net_profit_month': max(monthly, key=lambda r: r['net_profit_eur']),
            'leave_one_year_out': leave_out(yearly_s1), 'leave_one_month_out': leave_out(monthly),
            'interpretation_policy': 'Descriptive contributions and leave-one-period-out totals only; no dominance or quality threshold.'},
        'definitions': {'calendar': 'UTC entry calendar; completed-trade profit attributed entirely to entry year/month',
            'population': 'exactly existing MATCHED rows; no reassignment, warm-up or eligibility changes',
            'profit': 'existing ledger profit: net EUR cash including costs; decimal summation of stored values',
            'win_rate': 'wins/N, fraction 0..1; null for empty periods',
            'profit_contribution': 'signed yearly net / total S1 net; may be negative or exceed 100%; undefined if total net zero',
            'empty_period': 'N=0, net=0; WR and expectancy null; excluded from positive/negative/zero counts'}}
    return pd.DataFrame(yearly+monthly), summary


def run():
    targets = (OUTPUT/'h5_temporal_stability.csv', OUTPUT/'h5_temporal_stability_summary.json')
    if any(p.exists() for p in targets):
        raise FileExistsError('temporal stability outputs already exist; refusing overwrite')
    before = {str(p): sha256_file(p) for p in sorted(OUTPUT.glob('h5*')) if p.is_file()}
    original = json.loads(H5_SUMMARY.read_text(encoding='utf-8'))
    if before[str(LEDGER)] != original['provenance']['trades_csv_sha256']:
        raise ValueError('H5 ledger checksum mismatch')
    trades = pd.read_csv(LEDGER, dtype={'profit': str})
    table, summary = analyze(trades)
    for state, values in summary['overall_by_state'].items():
        old = original['by_state'][state]
        if any(values[k] != old[k] for k in ('N', 'wins', 'losses', 'net_profit_eur')):
            raise ValueError('existing H5 state aggregates do not reconcile')
    if any(sha256_file(Path(p)) != digest for p, digest in before.items()):
        raise RuntimeError('source H5 artifact changed')
    summary['provenance'] = {'source_sha256_before_and_after': before, 'model_calls': 0, 'feature_calls': 0}
    payloads = (table.to_csv(index=False, lineterminator='\n'),
                json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n')
    for target, payload in zip(targets, payloads):
        with target.open('x', encoding='utf-8', newline='') as stream:
            stream.write(payload)
    return summary


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('reconciliation', 'yearly_states', 'monthly_S1_stability',
                                             'expectancy_sign_year_counts', 'S1_concentration')}, indent=2))
