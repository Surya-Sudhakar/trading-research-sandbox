"""Read-only explanation of frozen H5 missing vectors; no inference or feature replay."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sandbox.provenance import sha256_file
from . import h5_discovery_2016_2022 as h5
from .validation import EXPECTED_MODEL_SHA256
from .validation_contract import validate_contract

OUTPUT = Path('results/market_state')
LEDGER = OUTPUT / 'h5_discovery_2016_2022_trades.csv'
SUMMARY = OUTPUT / 'h5_discovery_2016_2022_summary.json'
UNMATCHED = 'NO_COMPLETE_VECTOR_AT_BOUNDARY'
EXPECTED_COUNTS = (4162, 2384, 1778)


def _times(values):
    result = pd.DatetimeIndex(values)
    if result.tz is None or result.hasnans:
        raise ValueError('valid timezone-aware timestamps required')
    return result.tz_convert('UTC')


def reconcile(trades):
    counts = (len(trades), int(trades.state_match_status.eq('MATCHED').sum()),
              int(trades.state_match_status.eq(UNMATCHED).sum()))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'H5 reconciliation failed: {counts} != {EXPECTED_COUNTS}')
    matched = trades.state_match_status.eq('MATCHED')
    if not trades.loc[matched, 'state_at_entry'].isin(h5.STATES).all() or trades.loc[~matched, 'state_at_entry'].notna().any():
        raise ValueError('H5 state/status inconsistency')
    return dict(zip(('total_trades', 'matched_trades', 'unmatched_trades'), counts))


def diagnose(trades, features, metadata):
    """Pure diagnostic. Future timestamps are displayed only, never consumed by classification.

    Warm-up requirements come from the authenticated artifact's engine definitions,
    not the abbreviated state-feature schema lookback descriptions. Counts are
    completed bars, with partial UTC H1/H3 buckets excluded. No prices are calculated.
    """
    counts = reconcile(trades)
    entry = _times(trades.entry_time)
    required = _times(trades.required_state_timestamp)
    if not required.equals(entry.floor('15min')) or not ((entry >= h5.START) & (entry < h5.END)).all():
        raise ValueError('H5 entry/boundary outside discovery or inconsistent')
    f = features.copy().sort_values('timestamp', kind='stable').reset_index(drop=True)
    f['timestamp'] = _times(f.timestamp)
    if f.timestamp.duplicated().any() or not f.timestamp.between(h5.START, h5.END, inclusive='left').all():
        raise ValueError('unique discovery feature timestamps required')
    if not f.timestamp.eq(f.timestamp.dt.floor('15min')).all() or f.continuity_segment_id.isna().any():
        raise ValueError('invalid feature grid or segment identity')
    if tuple(c for c in f if c in h5.FEATURES) != h5.FEATURES:
        raise ValueError('frozen feature order mismatch')
    for key, value in (('provider', 'DUKASCOPY'), ('symbol', 'EURUSD'), ('decision_timeframe', 'M15')):
        if not f[key].eq(value).all():
            raise ValueError('feature source mismatch')
    if metadata['quality_mode'] != 'SEGMENT_LOCAL_V1' or metadata['source_timeframe'] != 'M15':
        raise ValueError('unsupported feature lineage')
    definitions = {d['name']: d for d in metadata['definitions']}
    requirements = {name: (definitions[name]['timeframe'], int(definitions[name]['lookback'])) for name in h5.FEATURES}
    if any(tf not in ('M15', 'H1', 'H3') or n <= 0 for tf, n in requirements.values()):
        raise ValueError('unsupported warm-up definition')
    # Metadata gaps refer to source opens; feature rows refer to closes.
    gaps = [g for g in metadata['decision_validation']['gaps']
            if h5.START <= pd.Timestamp(g['current']) < h5.END]
    gap_starts = {pd.Timestamp(g['current']): g for g in gaps}
    first_source = pd.Timestamp(metadata['decision_validation']['earliest'])
    runs = f.continuity_segment_id.ne(f.continuity_segment_id.shift()).cumsum()
    if f.groupby(runs).continuity_segment_id.first().duplicated().any():
        raise ValueError('noncontiguous segment identity')
    segment_info = {}
    causal_spacing_valid = np.ones(len(f), dtype=bool)
    for sid, group in f.groupby('continuity_segment_id', sort=False):
        opened = group.timestamp.iloc[0] - h5.STEP
        spacing = group.timestamp.diff().eq(h5.STEP).to_numpy()
        spacing[0] = True
        causal_spacing_valid[group.index] = np.logical_and.accumulate(spacing)
        documented = opened == first_source or opened in gap_starts
        segment_info[sid] = (opened, documented, gap_starts.get(opened))
    times = pd.DatetimeIndex(f.timestamp)
    finite = np.isfinite(f.loc[:, h5.FEATURES].to_numpy(float))
    complete_times = times[finite.all(axis=1)]
    complete_at_trade = required.isin(complete_times)
    if not np.array_equal(complete_at_trade, trades.state_match_status.eq('MATCHED').to_numpy()):
        raise ValueError('H5 ledger and frozen feature availability disagree')
    rows = []
    for trade_i in np.flatnonzero(trades.state_match_status.eq(UNMATCHED).to_numpy()):
        t = required[trade_i]
        position = times.searchsorted(t)
        exists = position < len(times) and times[position] == t
        row = dict(entry_time=entry[trade_i].isoformat(), required_state_timestamp=t.isoformat(),
                   weekday=t.day_name(), hour_utc=t.hour, missing_features=None,
                   number_missing_features=None, continuity_segment_id=None,
                   bars_since_segment_start=None, segment_start_utc=None,
                   completed_m15_bars=None, completed_h1_bars=None, completed_h3_bars=None,
                   warmup_shortfall_features=None, preceding_gap_classification=None,
                   diagnostic_reason='UNKNOWN', cause_explained=False)
        if exists:
            saved = f.iloc[position]
            sid = saved.continuity_segment_id
            opened, trustworthy, gap = segment_info[sid]
            trustworthy = trustworthy and causal_spacing_valid[position]
            missing = [name for name, valid in zip(h5.FEATURES, finite[position]) if not valid]
            row.update(missing_features=json.dumps(missing), number_missing_features=len(missing),
                       continuity_segment_id=str(sid), segment_start_utc=opened.isoformat(),
                       preceding_gap_classification=gap['classification'] if gap else None)
            if trustworthy:
                available = {'M15': int((t-opened)//h5.STEP)}
                for tf, hours in (('H1', 1), ('H3', 3)):
                    available[tf] = max(0, int((t-opened.ceil(f'{hours}h'))//timedelta(hours=hours)))
                short = [name for name in missing if available[requirements[name][0]] < requirements[name][1]]
                row.update(bars_since_segment_start=available['M15']-1,
                           completed_m15_bars=available['M15'], completed_h1_bars=available['H1'],
                           completed_h3_bars=available['H3'], warmup_shortfall_features=json.dumps(short))
                if len(short) == len(missing) and missing:
                    # Weekend-associated is a calendar description, not certification of a closure.
                    weekly = gap and pd.Timestamp(gap['previous']).weekday() == 4 and opened.weekday() in (6, 0)
                    reason = ('WEEKLY_SEGMENT_WARMUP' if weekly else
                              'MIDWEEK_SEGMENT_WARMUP' if gap else 'INITIAL_SEGMENT_WARMUP')
                    row.update(diagnostic_reason=reason, cause_explained=True)
                else:
                    row['diagnostic_reason'] = 'SPECIFIC_FEATURE_INCOMPLETE'
        else:
            source_open = t-h5.STEP
            in_gap = any(pd.Timestamp(g['previous']) < source_open < pd.Timestamp(g['current']) for g in gaps)
            row.update(diagnostic_reason='SOURCE_BAR_GAP' if in_gap else 'NO_FEATURE_ROW_AT_BOUNDARY',
                       cause_explained=bool(in_gap))
        # Display-only neighbors, added AFTER classification; never assign a state.
        row['previous_available_feature_timestamp'] = times[position-1].isoformat() if position else None
        next_position = position+1 if exists else position
        row['next_available_feature_timestamp'] = times[next_position].isoformat() if next_position < len(times) else None
        ci = complete_times.searchsorted(t)
        row['previous_complete_vector_timestamp'] = complete_times[ci-1].isoformat() if ci else None
        row['next_complete_vector_timestamp'] = complete_times[ci].isoformat() if ci < len(complete_times) else None
        rows.append(row)
    diagnostic = pd.DataFrame(rows)
    reasons = Counter(diagnostic.diagnostic_reason)
    missing_frequency = Counter(name for value in diagnostic.missing_features.dropna() for name in json.loads(value))
    dt = pd.to_datetime(diagnostic.required_state_timestamp, utc=True)
    explained = int(diagnostic.cause_explained.sum())
    def tally(values):
        return {str(k): int(v) for k, v in sorted(Counter(values).items())}
    summary = dict(experiment='H5_MISSING_STATE_DIAGNOSTIC_V1', **counts,
        by_diagnostic_reason={k: {'count': v, 'percent_of_unmatched': 100*v/len(rows)} for k, v in sorted(reasons.items())},
        by_year=tally(dt.dt.year), by_weekday=tally(diagnostic.weekday), by_hour_utc=tally(dt.dt.hour),
        missing_feature_frequency=dict(sorted(missing_frequency.items())),
        continuity={'feature_segments': len(segment_info), 'resets_observed': max(0, len(segment_info)-1),
                    'documented_gaps_in_discovery': len(gaps),
                    'segments_with_documented_start': sum(v[1] for v in segment_info.values()),
                    'unmatched_segments': int(diagnostic.continuity_segment_id.nunique()),
                    'unmatched_with_determined_segment_age': int(diagnostic.bars_since_segment_start.notna().sum())},
        monday_plus_tuesday_before_09_utc=int(((dt.dt.weekday == 0) | ((dt.dt.weekday == 1) & (dt.dt.hour < 9))).sum()),
        wednesday_friday_unexplained=int((dt.dt.weekday.between(2, 4) & ~diagnostic.cause_explained).sum()),
        explained_count=explained, unexplained_count=len(rows)-explained, unknown_count=reasons.get('UNKNOWN', 0),
        missingness_status='FULLY_EXPLAINED' if explained == len(rows) else 'PARTLY_EXPLAINED' if explained else 'UNRESOLVED',
        definitions={'bars_since_segment_start': 'zero-based completed M15 bar index within verified segment',
                     'weekday_hour_basis': 'required_state_timestamp UTC',
                     'neighbor_timestamps': 'strict previous/next stored row, and separate complete-vector neighbors; diagnostic only, never assigned',
                     'weekly': 'reset after Friday to Sunday/Monday; calendar association only, not certified market closure',
                     'unexplained': 'cause_explained=false, including feature-specific nulls without proven warm-up cause',
                     'warmup': 'all missing fields individually lack required completed bars according to frozen artifact definitions'},
        feature_requirements={k: {'timeframe': tf, 'completed_bars': n} for k, (tf, n) in requirements.items()})
    return diagnostic, summary


def run():
    targets = (OUTPUT/'h5_missing_state_diagnostic.csv', OUTPUT/'h5_missing_state_diagnostic_summary.json')
    if any(p.exists() for p in targets):
        raise FileExistsError('diagnostic already exists; refusing overwrite')
    contract = json.loads(h5.DEFAULT_CONTRACT.read_text(encoding='utf-8'))
    validate_contract(contract)
    feature_path = Path(contract['feature_artifact']['path'])
    metadata_path = feature_path.with_name('metadata.json')
    if sha256_file(metadata_path) != contract['feature_artifact']['metadata_sha256']:
        raise ValueError('feature metadata checksum mismatch')
    if sha256_file(h5.DEFAULT_MODEL) != EXPECTED_MODEL_SHA256:
        raise ValueError('frozen model checksum mismatch')
    protected = {p for p in OUTPUT.rglob('*') if p.is_file()} | {feature_path, metadata_path, h5.EA}
    before = {str(p): sha256_file(p) for p in sorted(protected)}
    h5_summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    if (before[str(LEDGER)] != h5_summary['provenance']['trades_csv_sha256'] or
            h5_summary['provenance']['feature_artifact'] != contract['feature_artifact']):
        raise ValueError('H5 provenance mismatch')
    with h5.no_fitting() as guard:
        trades = pd.read_csv(LEDGER, usecols=['entry_time', 'required_state_timestamp', 'state_at_entry', 'state_match_status'])
        features = h5.read_features(feature_path, contract)
        diagnostic, summary = diagnose(trades, features, json.loads(metadata_path.read_text(encoding='utf-8')))
    if guard['fit_calls'] or any(sha256_file(Path(p)) != digest for p, digest in before.items()):
        raise RuntimeError('read-only diagnostic invariant violated')
    summary['provenance'] = {'protected_sha256_before_and_after': before, 'fit_calls': guard['fit_calls'],
                             'feature_loading': 'existing H5 read_features; predicate-filtered 2016 <= timestamp < 2023',
                             'inference_calls': 0, 'feature_recalculations': 0}
    payloads = (diagnostic.to_csv(index=False, lineterminator='\n'), json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n')
    for target, payload in zip(targets, payloads):
        with target.open('x', encoding='utf-8', newline='') as stream:
            stream.write(payload)
    return summary


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('total_trades', 'matched_trades', 'unmatched_trades',
          'by_diagnostic_reason', 'missing_feature_frequency', 'missingness_status', 'unknown_count')}, indent=2))
