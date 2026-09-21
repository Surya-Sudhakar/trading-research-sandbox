"""Descriptive within-state H6 comparisons; reads frozen ledger fields only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sandbox.provenance import sha256_file

OUTPUT = Path('results/market_state')
LEDGER = OUTPUT/'h5_discovery_2016_2022_trades.csv'
H5_SUMMARY = OUTPUT/'h5_discovery_2016_2022_summary.json'
VARIABLES = ('state_age', 'bars_since_transition', 'max_posterior', 'posterior_margin', 'posterior_entropy')
INPUT_COLUMNS = ('entry_time', 'required_state_timestamp', 'state_timestamp', 'state_match_status',
                 'state_at_entry', 'outcome', *VARIABLES)
STATES = {'S0': 114, 'S1': 794, 'S2': 1476}


def distribution(values):
    valid = values.dropna().to_numpy(float)
    q = np.quantile(valid, [.25, .5, .75]) if len(valid) else [None]*3
    return {'N': len(valid), 'total': len(values), 'missing': int(values.isna().sum()),
            'mean': float(valid.mean()) if len(valid) else None, 'median': float(q[1]) if len(valid) else None,
            'q25': float(q[0]) if len(valid) else None, 'q75': float(q[2]) if len(valid) else None,
            'IQR': float(q[2]-q[0]) if len(valid) else None}


def effects(wins, losses):
    """Loss minus win. Cliff's delta includes ties; g uses pooled sample SD."""
    w, l = wins.dropna().to_numpy(float), losses.dropna().to_numpy(float)
    result = {'mean_difference': None, 'median_difference': None, 'cliffs_delta': None, 'hedges_g': None}
    if not len(w) or not len(l):
        return result
    sorted_w = np.sort(w)
    greater = np.searchsorted(sorted_w, l, side='left').sum()
    smaller = (len(w)-np.searchsorted(sorted_w, l, side='right')).sum()
    result.update(mean_difference=float(l.mean()-w.mean()), median_difference=float(np.median(l)-np.median(w)),
                  cliffs_delta=float((greater-smaller)/(len(w)*len(l))))
    if len(w) >= 2 and len(l) >= 2:
        df = len(w)+len(l)-2
        pooled = np.sqrt(((len(w)-1)*w.var(ddof=1)+(len(l)-1)*l.var(ddof=1))/df)
        if pooled > 0:
            result['hedges_g'] = float((1-3/(4*df-1))*(l.mean()-w.mean())/pooled)
    return result


def analyze(ledger):
    # Explicit projection: prices, exits, profit and any other future fields cannot enter predictors.
    source = ledger.loc[:, INPUT_COLUMNS].copy()
    entry = pd.DatetimeIndex(source.entry_time)
    if entry.tz is None or entry.hasnans:
        raise ValueError('aware entry times required')
    entry = entry.tz_convert('UTC')
    if not ((entry >= pd.Timestamp('2016-01-01', tz='UTC')) & (entry < pd.Timestamp('2023-01-01', tz='UTC'))).all():
        raise ValueError('discovery 2016-2022 only')
    matched_mask = source.state_match_status.eq('MATCHED')
    if (len(source), int(matched_mask.sum()), int(source.state_match_status.eq('NO_COMPLETE_VECTOR_AT_BOUNDARY').sum())) != (4162, 2384, 1778):
        raise ValueError('H5 population mismatch')
    matched = source.loc[matched_mask].copy()
    matched['entry_time'] = entry[matched_mask]
    if matched.state_at_entry.value_counts().to_dict() != STATES:
        raise ValueError('state counts mismatch')
    required = pd.DatetimeIndex(matched.required_state_timestamp)
    actual = pd.DatetimeIndex(matched.state_timestamp)
    if (required.tz is None or actual.tz is None or required.hasnans or actual.hasnans or
            not required.tz_convert('UTC').equals(pd.DatetimeIndex(matched.entry_time).floor('15min')) or
            not actual.tz_convert('UTC').equals(required.tz_convert('UTC'))):
        raise ValueError('noncausal or inconsistent state boundary')
    if not matched.outcome.isin(['WIN', 'LOSS']).all():
        raise ValueError('invalid outcomes')
    for name in VARIABLES:
        matched[name] = pd.to_numeric(matched[name], errors='raise')
        v = matched[name].dropna()
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError('invalid pre-entry variable')
        if name in ('max_posterior', 'posterior_margin') and (v > 1).any():
            raise ValueError('invalid probability variable')
    rows = []
    for period in ('ALL', *map(str, range(2016, 2023))):
        frame = matched if period == 'ALL' else matched.loc[matched.entry_time.dt.year.eq(int(period))]
        for state in STATES:
            group = frame.loc[frame.state_at_entry.eq(state)]
            for name in VARIABLES:
                w = group.loc[group.outcome.eq('WIN'), name]
                l = group.loc[group.outcome.eq('LOSS'), name]
                row = {'period': period, 'state': state, 'variable': name,
                       **{f'WIN_{k}': v for k, v in distribution(w).items()},
                       **{f'LOSS_{k}': v for k, v in distribution(l).items()}, **effects(w, l)}
                rows.append(row)
    temporal = []
    for state in STATES:
        for name in VARIABLES:
            subset = [r for r in rows if r['period'] != 'ALL' and r['state'] == state and r['variable'] == name]
            temporal.append({'state': state, 'variable': name,
                'years_with_both_groups': sum(r['cliffs_delta'] is not None for r in subset),
                'years_with_at_least_two_per_group': sum(r['WIN_N'] >= 2 and r['LOSS_N'] >= 2 for r in subset),
                'positive_delta_years': [r['period'] for r in subset if r['cliffs_delta'] is not None and r['cliffs_delta'] > 0],
                'negative_delta_years': [r['period'] for r in subset if r['cliffs_delta'] is not None and r['cliffs_delta'] < 0],
                'zero_delta_years': [r['period'] for r in subset if r['cliffs_delta'] == 0]})
    summary = {'experiment': 'H6_STATE_STABILITY_DISCOVERY_V1',
        'reconciliation': {'total_ledger': len(source), 'matched': len(matched), 'excluded_unmatched': 1778,
                           'state_counts': STATES},
        'state_outcomes': {s: {o: int(((matched.state_at_entry == s) & (matched.outcome == o)).sum())
                               for o in ('WIN', 'LOSS')} for s in STATES},
        'predictors': list(VARIABLES),
        'pooled_matched_reference_quantiles': {name: distribution(matched[name]) for name in VARIABLES},
        'comparisons': [r for r in rows if r['period'] == 'ALL'],
        'yearly_comparisons': [r for r in rows if r['period'] != 'ALL'], 'temporal_consistency': temporal,
        'definitions': {'orientation': 'all effect sizes and differences are LOSS minus WIN',
            'cliffs_delta': 'P(loss value > win value) - P(loss value < win value); ties contribute zero',
            'hedges_g': 'pooled-sample-SD standardized mean difference times 1-3/(4*(n_loss+n_win-2)-1); null if either n<2 or pooled SD=0',
            'IQR': 'linear-interpolated q75-q25; quantiles calculated on observed values only',
            'missingness': 'per-variable available-case summaries; no imputation or population re-eligibility',
            'reference_quantiles': 'full matched population, no outcome conditioning, no binning or cutoff selection',
            'state_age': 'existing one-based consecutive-state age, resets at gaps/missing vectors; left-censored',
            'bars_since_transition': 'existing zero-based age since observed change, unknown until change after a reset',
            'temporal': 'UTC entry year; all years retained; singleton-loss comparisons descriptive, no sample SD estimate'},
        'limitations': ['S1 has only 9 losses versus 785 wins; comparisons are unstable and not confirmatory.',
            'Trade observations may be temporally dependent; no independent-sample significance claims or p-values.',
            'This is in-sample discovery; correlated predictors and multiple descriptive comparisons are not independent evidence.',
            'Within-state comparisons describe marginal separation, not incremental predictive utility or a multivariable adjustment.',
            'Unknown transition ages are retained as missing; available-case transition comparisons can be selective.'],
        'operations': {'model_calls': 0, 'fit_calls': 0, 'threshold_searches': 0,
                       'feature_calculations': 0, 'state_assignments': 0}}
    return pd.DataFrame(rows), summary


def run():
    targets = (OUTPUT/'h6_state_stability_discovery.csv', OUTPUT/'h6_state_stability_discovery_summary.json')
    if any(p.exists() for p in targets):
        raise FileExistsError('H6 output exists; refusing overwrite')
    protected = set(OUTPUT.glob('h5*')) | set(OUTPUT.glob('state_transition_discovery_v1*'))
    before = {str(p): sha256_file(p) for p in sorted(protected) if p.is_file()}
    old = json.loads(H5_SUMMARY.read_text(encoding='utf-8'))
    if before[str(LEDGER)] != old['provenance']['trades_csv_sha256']:
        raise ValueError('H5 ledger checksum mismatch')
    table, summary = analyze(pd.read_csv(LEDGER, usecols=list(INPUT_COLUMNS)))
    for state in STATES:
        if summary['state_outcomes'][state] != {'WIN': old['by_state'][state]['wins'], 'LOSS': old['by_state'][state]['losses']}:
            raise ValueError('frozen H5 outcome counts differ')
    if any(sha256_file(Path(p)) != digest for p, digest in before.items()):
        raise RuntimeError('H5/H4 artifact changed')
    summary['provenance'] = {'protected_sha256_before_and_after': before, 'loaded_columns': list(INPUT_COLUMNS)}
    for target, payload in zip(targets, (table.to_csv(index=False, lineterminator='\n'),
                              json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n')):
        with target.open('x', encoding='utf-8', newline='') as stream:
            stream.write(payload)
    return summary


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('reconciliation', 'state_outcomes', 'comparisons', 'temporal_consistency')}, indent=2))
