"""Explicitly authorized August 2026 cross-feed diagnostic. Inference only, never H5."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sandbox.provenance import sha256_file
from sandbox.market_state.universal.engine import UniversalFeatureEngine
from sandbox.market_state.universal.continuity import QualityContext
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1 as FEATURES
from .validation import load_frozen_inputs, no_fitting, DEFAULT_MODEL, DEFAULT_CONTRACT, EXPECTED_MODEL_SHA256
from .validation_contract import CONTRACT_SHA256

UTC = timezone.utc
START = datetime(2026, 8, 3, tzinfo=UTC)
END = datetime(2026, 8, 9, tzinfo=UTC)
STEP = timedelta(minutes=15)
STATES = ('S0', 'S1', 'S2')
POSTERIORS = ('p_S0', 'p_S1', 'p_S2', 'max_posterior', 'posterior_entropy', 'posterior_margin')
OUTPUT = Path('results/market_state')


def ticks(path, feed, stats):
    """Stream in original order; IC quote state is updated solely by current/past ticks."""
    if feed not in ('duka', 'ic'):
        raise ValueError('unknown feed')
    bid = ask = previous = None
    stats.update(raw_rows=0, outside_authorized_window=0, bid_unavailable=0, yielded_bid_ticks=0,
                 duplicate_timestamps=0, crossed_quote_ticks=0)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=',' if feed == 'duka' else '\t')
        required = {'timestamp', 'askPrice', 'bidPrice'} if feed == 'duka' else {'<DATE>', '<TIME>', '<BID>', '<ASK>', '<FLAGS>'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('tick header mismatch')
        for row in reader:
            stats['raw_rows'] += 1
            if feed == 'duka':
                stamp = datetime(1970, 1, 1, tzinfo=UTC)+timedelta(milliseconds=int(row['timestamp']))
            else:
                stamp = datetime.fromisoformat(row['<DATE>'].replace('.', '-')+'T'+row['<TIME>']).replace(tzinfo=UTC)-timedelta(hours=3)
            # No out-of-scope quotes enter aggregation or feature computation.
            if not START <= stamp < END:
                stats['outside_authorized_window'] += 1
                continue
            if previous is not None and stamp < previous:
                raise ValueError('backwards source timestamps')
            stats['duplicate_timestamps'] += int(stamp == previous)
            previous = stamp
            def quote(raw):
                if raw in ('', None):
                    return None
                value = float(raw)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('invalid quote')
                return value
            if feed == 'duka':
                bid, ask = quote(row['bidPrice']), quote(row['askPrice'])
                if bid is None or ask is None:
                    raise ValueError('Dukascopy quote incomplete')
            else:
                flags = int(row['<FLAGS>'])
                if flags < 0:
                    raise ValueError('invalid flags')
                for side, flag in (('bid', 2), ('ask', 4)):
                    value = quote(row[f'<{side.upper()}>'])
                    old = bid if side == 'bid' else ask
                    if flags & flag and value is None:
                        raise ValueError('flagged quote update missing')
                    if value is not None:
                        if not flags & flag and old is not None and value != old:
                            raise ValueError('unflagged quote change')
                        if side == 'bid': bid = value
                        else: ask = value
            if bid is None:
                stats['bid_unavailable'] += 1
                continue
            stats['crossed_quote_ticks'] += int(ask is not None and ask < bid)
            stats['yielded_bid_ticks'] += 1
            yield stamp, bid, ask


def bounds(path, feed):
    stats = {}
    first = last = None
    for stamp, _, _ in ticks(path, feed, stats):
        if first is None: first = stamp
        last = stamp
    if first is None:
        raise ValueError('no eligible bid ticks')
    return first, last, stats


def aggregate(path, feed, overlap_start, overlap_end):
    """Common interval only; partial first/last M15 buckets excluded, no empty bars."""
    if not START <= overlap_start <= overlap_end < END:
        raise ValueError('overlap outside authorized diagnostic window')
    first_open = pd.Timestamp(overlap_start).ceil('15min').to_pydatetime()
    last_close = pd.Timestamp(overlap_end).floor('15min').to_pydatetime()
    buckets, stats = {}, {}
    included = partial = 0
    for stamp, bid, _ in ticks(path, feed, stats):
        if stamp < overlap_start or stamp > overlap_end:
            continue
        if stamp < first_open or stamp >= last_close:
            partial += 1
            continue
        opened = stamp.replace(minute=stamp.minute//15*15, second=0, microsecond=0)
        if opened not in buckets:
            buckets[opened] = [bid, bid, bid, bid, 1]
        else:
            b = buckets[opened]
            b[1], b[2], b[3], b[4] = max(b[1], bid), min(b[2], bid), bid, b[4]+1
        included += 1
    frame = pd.DataFrame([(t, *v) for t, v in buckets.items()], columns=['timestamp_utc', 'open', 'high', 'low', 'close', 'observed_ticks'])
    frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame['continuity_segment_id'] = frame.timestamp_utc.diff().ne(STEP).cumsum()-1
    grid_count = max(0, int((last_close-first_open)/STEP))
    return frame, {'m15_bars': len(frame), 'complete_interval_grid_slots': grid_count,
                   'empty_m15_slots_not_filled': grid_count-len(frame),
                   'ticks_used': included, 'partial_boundary_ticks_excluded': partial,
                   'first_complete_bucket_open': first_open.isoformat(), 'complete_bucket_end_exclusive': last_close.isoformat()}


def infer(bars, model):
    """Reuse the canonical engine, segment-local and unchanged; leave unavailable vectors null."""
    columns = ['timestamp', 'continuity_segment_id', *FEATURES, 'unavailable_features', 'state', *POSTERIORS]
    if bars.empty:
        return pd.DataFrame({c: pd.Series(dtype='datetime64[ns, UTC]' if c == 'timestamp' else object) for c in columns})
    rows = UniversalFeatureEngine().compute(bars, symbol='EURUSD', quality_context=QualityContext())
    f = pd.DataFrame([{'timestamp': r.timestamp, **{name: getattr(r, name) for name in FEATURES}} for r in rows])
    f['continuity_segment_id'] = bars.continuity_segment_id.to_numpy()
    x = f.loc[:, FEATURES].to_numpy(float)
    ready = np.isfinite(x).all(axis=1)
    f['unavailable_features'] = [';'.join(name for name, value in zip(FEATURES, row) if not np.isfinite(value)) for row in x]
    f['state'] = pd.Series([None]*len(f), dtype=object)
    for name in POSTERIORS: f[name] = np.nan
    if ready.any():
        with no_fitting():
            p = model.predict_proba(x[ready], feature_names=FEATURES)
            labels = model.predict(x[ready], feature_names=FEATURES)
        if (p.shape != (int(ready.sum()), 3) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12) or not np.array_equal(labels, p.argmax(axis=1))):
            raise ValueError('invalid frozen predictions')
        logp = np.zeros_like(p)
        np.log(p, out=logp, where=p > 0)
        ordered = np.sort(p, axis=1)
        f.loc[ready, 'state'] = [STATES[int(i)] for i in labels]
        f.loc[ready, list(POSTERIORS)] = np.column_stack([p, p.max(axis=1), -(p*logp).sum(axis=1), ordered[:, 2]-ordered[:, 1]])
    return f.loc[:, columns]


def join_states(duka, ic):
    def named(frame, prefix):
        return frame.rename(columns={c: f'{prefix}_{c}' for c in frame if c != 'timestamp'})
    joined = named(duka, 'duka').merge(named(ic, 'ic'), on='timestamp', how='outer', validate='one_to_one', indicator=True)
    joined = joined.sort_values('timestamp', kind='stable').reset_index(drop=True)
    joined['matched_valid'] = joined.duka_state.notna() & joined.ic_state.notna()
    joined['agreement'] = pd.array([bool(a == b) if valid else None for a, b, valid in
        zip(joined.duka_state, joined.ic_state, joined.matched_valid)], dtype='boolean')
    return joined.rename(columns={'_merge': 'feed_presence'})


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
            'median': float(np.median(x)) if len(x) else None, 'p95': float(np.quantile(x, .95)) if len(x) else None}


def transitions(matched):
    connected = matched.timestamp.diff().eq(STEP)
    for feed in ('duka', 'ic'):
        connected &= matched[f'{feed}_continuity_segment_id'].eq(matched[f'{feed}_continuity_segment_id'].shift())
    connected = connected.to_numpy()
    events = {}
    result = {'consecutive_valid_edges': int(connected.sum()), 'disconnected_edges_excluded': max(0, len(matched)-1-int(connected.sum()))}
    for feed in ('duka', 'ic'):
        labels = matched[f'{feed}_state'].to_numpy()
        changed = np.r_[False, labels[1:] != labels[:-1]] if len(labels) else np.array([], dtype=bool)
        events[feed] = changed[connected]
        matrix = np.zeros((3, 3), dtype=int)
        for i in np.flatnonzero(connected):
            matrix[STATES.index(labels[i-1]), STATES.index(labels[i])] += 1
        runs = []
        for i in range(len(labels)):
            if i == 0 or not connected[i] or changed[i]: runs.append(1)
            else: runs[-1] += 1
        result[feed] = {'transition_counts': matrix.tolist(), 'state_changes': int(events[feed].sum()),
                         'self_transitions': int((~events[feed]).sum()), 'observed_run_bars': stats(runs),
                         'self_transition_fraction': float((~events[feed]).mean()) if len(events[feed]) else None}
    n = len(events['duka'])
    either = events['duka'] | events['ic']
    result['change_or_stay_agreement_percent'] = float(100*np.mean(events['duka'] == events['ic'])) if n else None
    result['both_change'] = int((events['duka'] & events['ic']).sum())
    result['duka_only_change'] = int((events['duka'] & ~events['ic']).sum())
    result['ic_only_change'] = int((~events['duka'] & events['ic']).sum())
    result['both_stay'] = int((~events['duka'] & ~events['ic']).sum())
    result['change_event_jaccard'] = float((events['duka'] & events['ic']).sum()/either.sum()) if either.any() else None
    return result


def compare(joined):
    m = joined.loc[joined.matched_valid].copy()
    matrix = np.zeros((3, 3), dtype=int)
    for a, b in zip(m.duka_state, m.ic_state): matrix[STATES.index(a), STATES.index(b)] += 1
    result = {'matched_state_count': len(m), 'total_common_m15_bars': int(joined.feed_presence.eq('both').sum()),
        'overall_agreement_percent': float(100*m.agreement.mean()) if len(m) else None,
        'confusion_matrix_rows_duka_columns_ic': matrix.tolist(),
        'per_state_agreement': {s: {'duka_reference_count': int(matrix[i].sum()),
            'percent_matching_ic_given_duka': float(100*matrix[i, i]/matrix[i].sum()) if matrix[i].sum() else None}
            for i, s in enumerate(STATES)},
        'coverage': {}, 'state_distributions': {},
        'posterior_absolute_differences': {p: stats(abs(m[f'duka_{p}']-m[f'ic_{p}'])) for p in POSTERIORS},
        'confidence_comparison_without_cutoff': {}, 'transitions': transitions(m)}
    for feed in ('duka', 'ic'):
        present = joined[f'{feed}_unavailable_features'].notna()
        result['coverage'][feed] = {'bars': int(present.sum()), 'valid_feature_rows': int(joined[f'{feed}_state'].notna().sum()),
            'unavailable_feature_rows': int((present & joined[f'{feed}_state'].isna()).sum()),
            'absent_bars_on_union': int((~present).sum()),
            'unavailable_by_feature': {name: int(joined.loc[present, f'{feed}_{name}'].isna().sum()) for name in FEATURES}}
        result['state_distributions'][feed] = {'all_valid': {s: int(joined[f'{feed}_state'].eq(s).sum()) for s in STATES},
                                               'matched_only': {s: int(m[f'{feed}_state'].eq(s).sum()) for s in STATES}}
        result['confidence_comparison_without_cutoff'][feed] = {
            group: {p: stats(m.loc[m.agreement.eq(agree), f'{feed}_{p}']) for p in ('max_posterior', 'posterior_entropy', 'posterior_margin')}
            for group, agree in (('agree', True), ('disagree', False))}
    result['unavailable_or_unmatched_union_rows'] = int((~joined.matched_valid).sum())
    return result


def run(duka_path, ic_path, output=OUTPUT):
    paths = {'duka': Path(duka_path), 'ic': Path(ic_path)}
    # Missing inputs stop before any observation is opened or output is published.
    for p in paths.values():
        if not p.is_file(): raise FileNotFoundError(p)
    output = Path(output)
    names = ('cross_feed_aug2026.csv', 'cross_feed_aug2026_disagreements.csv', 'cross_feed_aug2026_summary.json')
    if any((output/n).exists() for n in names): raise FileExistsError('diagnostic output already exists')
    protected = {p for p in OUTPUT.rglob('*') if p.is_file()} | {DEFAULT_MODEL, DEFAULT_CONTRACT}
    protected |= {p for p in Path('reference_strategies').rglob('*') if p.is_file()}
    protected |= {Path('sandbox/partition/service.py')}
    before = {str(p): sha256_file(p) for p in sorted(protected)}
    input_hashes = {feed: sha256_file(p) for feed, p in paths.items()}
    with no_fitting() as guard:
        contract, model, contract_file_sha = load_frozen_inputs()
        limits = {feed: bounds(p, feed) for feed, p in paths.items()}
        start = max(x[0] for x in limits.values())
        end = min(x[1] for x in limits.values())
        if start >= end: raise ValueError('no genuine overlap')
        bars, aggregation, assigned = {}, {}, {}
        for feed, p in paths.items():
            bars[feed], aggregation[feed] = aggregate(p, feed, start, end)
            assigned[feed] = infer(bars[feed], model)
        joined = join_states(assigned['duka'], assigned['ic'])
        summary = compare(joined)
    after = {p: sha256_file(Path(p)) for p in before}
    if before != after or any(sha256_file(paths[k]) != v for k, v in input_hashes.items()):
        raise ValueError('protected inputs changed')
    summary.update(experiment='CROSS_FEED_FROZEN_MARKET_STATE_AUG2026_DIAGNOSTIC_V1',
        diagnostic_only=True, fit_calls=guard['fit_calls'], model_sha256=EXPECTED_MODEL_SHA256,
        contract_checksum=CONTRACT_SHA256, contract_file_sha256=contract_file_sha,
        input_paths={k: str(p) for k, p in paths.items()}, input_sha256=input_hashes,
        exact_utc_overlap={'start_inclusive': start.isoformat(), 'last_common_coverage_timestamp': end.isoformat()},
        tick_inventory={k: v[2] for k, v in limits.items()}, aggregation=aggregation,
        feature_names=list(FEATURES), feature_schema_version=contract['feature_schema']['version'],
        protected_before=before, protected_after=after,
        rules={'timestamps': 'IC server clock minus 3 hours; Dukascopy epoch milliseconds UTC.',
            'candles': 'Bid OHLC; duplicate timestamps retain source order; quote carry is causal; no empty candle fill.',
            'warmup': 'Only fully contained common-interval M15 buckets; reset at each absent M15 bucket independently per feed. No pre-overlap feature warm-up.',
            'continuity': 'Use existing QualityContext and unchanged UniversalFeatureEngine; no policy or schema modification.',
            'joins': 'Exact feature decision-close timestamps, complete vectors on both feeds only.',
            'per_state_agreement': 'P(IC state equals Dukascopy state | Dukascopy state), on matched rows.',
            'boundaries': 'Agreement/disagreement confidence distributions only; no low-confidence threshold.',
            'transitions': 'Adjacent matched rows separated by exactly 15 minutes and unchanged segment IDs on BOTH feeds.'},
        limitations=['Sealed-period diagnostic only; no training, thresholds, model selection, H5 or performance analysis.',
            'Sparse/no-tick intervals cannot certify feed completeness. Empty M15 buckets are excluded and break history.',
            'Warm-up and gaps select the matched subset. Agreement is conditional on complete vectors.',
            'State IDs are frozen component indices; labels are never aligned or reordered.'])
    disagreements = joined.loc[joined.matched_valid & joined.agreement.eq(False)].copy()
    csvs = [joined.to_csv(index=False, lineterminator='\n').encode(), disagreements.to_csv(index=False, lineterminator='\n').encode()]
    summary['output_csv_sha256'] = {n: hashlib.sha256(b).hexdigest() for n, b in zip(names, csvs)}
    payloads = [*csvs, (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()]
    output.mkdir(parents=True, exist_ok=True)
    for name, data in zip(names, payloads):
        with (output/name).open('xb') as stream: stream.write(data)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duka', type=Path, required=True)
    parser.add_argument('--ic', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.duka, args.ic)
    print(json.dumps({k: result[k] for k in ('matched_state_count', 'overall_agreement_percent', 'confusion_matrix_rows_duka_columns_ic', 'fit_calls')}, indent=2))
