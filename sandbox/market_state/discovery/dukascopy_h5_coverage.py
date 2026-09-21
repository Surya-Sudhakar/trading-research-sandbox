"""Frozen-model pre-2025 tick coverage; no Version 0 execution or model development."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

import pandas as pd

from sandbox.market_data.dukascopy_csv import aggregate_csv
from sandbox.market_data.gaps import inventory, SessionProfile, ContinuityPolicy
from sandbox.provenance import sha256_file
from .cross_feed_diagnostic import infer, FEATURES, STATES
from .validation import no_fitting, load_frozen_inputs, DEFAULT_MODEL, DEFAULT_CONTRACT, EXPECTED_MODEL_SHA256
from .validation_contract import CONTRACT_SHA256

OUTPUT = Path('results/market_state/dukascopy_h5_feature_coverage.json')
CACHE = Path('results/market_state/dukascopy_h5_coverage_sources')


def publish(path, payload):
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes() != data: raise ValueError('refusing changed coverage artifact')
        return
    import os, tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name); stream.write(data); stream.flush(); os.fsync(stream.fileno())
    try: os.link(temporary, path)
    finally: temporary.unlink()


def frozen_continuity(contract):
    path = Path(contract['feature_artifact']['path']).with_name('metadata.json')
    if sha256_file(path) != contract['feature_artifact']['metadata_sha256']:
        raise ValueError('frozen feature metadata checksum mismatch')
    feature_meta = json.loads(path.read_text())
    with sqlite3.connect('file:data/catalog.sqlite3?mode=ro', uri=True) as db:
        row = db.execute('SELECT manifest_json FROM market_data_lineage WHERE dataset_id=?',
                         (feature_meta['source_dataset_id'],)).fetchone()
    metadata = json.loads(row[0])
    policy = ContinuityPolicy(**metadata['continuity_policy'])
    profile = SessionProfile(**metadata['session_profile'])
    if feature_meta['quality_mode'] != 'SEGMENT_LOCAL_V1': raise ValueError('unexpected training continuity mode')
    return profile, policy, {'quality_mode': feature_meta['quality_mode'],
        'policy': metadata['continuity_policy'], 'session_profile': metadata['session_profile'],
        'evidence': ['sandbox/market_data/gaps.py: SessionProfile.classify and inventory',
            'sandbox/market_state/universal/dataset.py: export_discovery_features',
            'sandbox/market_state/universal/continuity.py: QualityContext.split',
            'sandbox/market_state/universal/engine.py: compute',
            'tests/test_multisource_data.py: test_continuity_resets_unexplained_and_can_continue_expected_weekend'],
        'interpretation': 'Verified EXPECTED_WEEKEND can preserve history. Original training used UNKNOWN_PROVIDER_SESSION; weekend-associated unknown gaps reset when missing duration >=60 minutes. Not every possible gap resets under CONTINUITY_V1.',
        'short_gap_safety': 'Any frozen-policy segment with an unexplained non-reset gap is quarantined from feature generation, never bridged or silently reclassified.'}


def source_coverage(path, model, profile, policy):
    candles, months = aggregate_csv(path)
    full = candles.loc[~candles.partial_boundary].copy().reset_index(drop=True)
    if full.empty: raise ValueError('no complete M15 buckets')
    gaps, ids = inventory(full, SimpleNamespace(timeframe='M15', provider='DUKASCOPY', symbol='EURUSD'), profile, policy)
    full['continuity_segment_id'] = ids
    unsafe = set()
    if len(gaps):
        unsafe = set(gaps.loc[~gaps.reset & gaps.classification.ne('EXPECTED_WEEKEND'), 'next_segment_id'])
    safe = full.loc[~full.continuity_segment_id.isin(unsafe)].copy()
    assigned = infer(safe, model)
    # Report by raw month; feature assignment is not restarted at month boundaries.
    for month, record in months.items():
        start = pd.Timestamp(month+'-01', tz='UTC'); end = start+pd.offsets.MonthBegin(1)
        bars = full.loc[full.timestamp_utc.between(start, end, inclusive='left')]
        rows = assigned.loc[assigned.timestamp.between(start, end, inclusive='left')]
        valid = rows.loc[rows.state.notna()]
        missing_slots = weekday_slots = closure_slots = unknown_weekend_slots = 0
        gap_events = []
        for gap in gaps.to_dict('records') if len(gaps) else []:
            left = max(pd.Timestamp(gap['previous_timestamp'])+timedelta(minutes=15), start)
            right = min(pd.Timestamp(gap['next_timestamp']), end)
            if left >= right: continue
            slots = int((right-left)/timedelta(minutes=15)); missing_slots += slots
            if gap['classification'] == 'EXPECTED_WEEKEND': closure_slots += slots
            elif gap['classification'] == 'WEEKEND_ASSOCIATED_UNKNOWN': unknown_weekend_slots += slots
            else: weekday_slots += slots
            gap_events.append({**gap, 'month_missing_slots': slots})
        record.update(m15_candles=len(bars), missing_m15_intervals=missing_slots,
            unexplained_gap_missing_slots=weekday_slots, verified_closure_missing_slots=closure_slots,
            unverified_weekend_missing_slots=unknown_weekend_slots, gaps=gap_events,
            partial_boundary_candles=int((candles.partial_boundary & candles.timestamp_utc.between(start,end,inclusive='left')).sum()),
            quarantined_m15_candles=int(bars.continuity_segment_id.isin(unsafe).sum()),
            feature_rows=len(rows), complete_feature_vectors=len(valid),
            first_complete_vector_timestamp=valid.timestamp.iloc[0].isoformat() if len(valid) else None,
            state_counts={s:int(valid.state.eq(s).sum()) for s in STATES},
            unavailable_by_feature={name:int(rows[name].isna().sum()) for name in FEATURES})
    return {'months': months, 'raw_ticks': sum(m['raw_tick_count'] for m in months.values()),
        'complete_m15_candles': len(full), 'partial_boundary_candles': int(candles.partial_boundary.sum()),
        'quarantined_continuity_segments': sorted(int(x) for x in unsafe),
        'complete_vectors': int(assigned.state.notna().sum()), 'feature_names': list(FEATURES),
        'first_tick': next(iter(months.values()))['first_timestamp'], 'last_tick': list(months.values())[-1]['last_timestamp']}


def discover(roots):
    found = []
    pattern = re.compile(r'eurusd-tick-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})\.csv', re.I)
    for root in roots:
        for p in Path(root).rglob('*.csv'):
            match = pattern.fullmatch(p.name)
            if match and '2016-01-01' <= match[1] < match[2] <= '2025-01-01': found.append(p)
    return sorted(set(found))


def run(roots):
    if OUTPUT.exists(): raise FileExistsError('coverage report exists; preserve existing research')
    files = discover(roots)
    if not files: raise ValueError('no eligible pre-2025 dukascopy-node CSVs')
    protected = {p:sha256_file(p) for p in Path('results/market_state').rglob('*') if p.is_file()}
    protected.update({p:sha256_file(p) for p in Path('reference_strategies').rglob('*') if p.is_file()})
    records, failures, aliases, hashes = [], [], [], {}
    with no_fitting() as guard:
        contract, model, contract_file_sha = load_frozen_inputs()
        profile, policy, evidence = frozen_continuity(contract)
        for path in files:
            digest = sha256_file(path)
            if digest in hashes:
                aliases.append({'path':str(path),'identical_source':hashes[digest],'sha256':digest}); continue
            hashes[digest] = str(path)
            cache = CACHE/(digest+'.json')
            print(json.dumps({'processing':str(path),'sha256':digest}),flush=True)
            try:
                if cache.exists():
                    report = json.loads(cache.read_text())
                    if report['source_sha256'] != digest or report['model_sha256'] != EXPECTED_MODEL_SHA256 or report['continuity'] != evidence:
                        raise ValueError('cached coverage binding mismatch')
                else:
                    report = source_coverage(path, model, profile, policy)
                    report.update(source_path=str(path),source_sha256=digest,model_sha256=EXPECTED_MODEL_SHA256,continuity=evidence)
                    if sha256_file(path) != digest: raise ValueError('source changed during processing')
                    publish(cache, report)
                records.append(report)
                print(json.dumps({'completed':str(path),'ticks':report['raw_ticks'],'complete_vectors':report['complete_vectors']}),flush=True)
            except (ValueError, pd.errors.ParserError) as exc:
                failures.append({'path':str(path),'sha256':digest,'error':str(exc)})
                print(json.dumps({'failed':str(path),'error':str(exc)}),flush=True)
    overlaps=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if max(a['first_tick'],b['first_tick']) <= min(a['last_tick'],b['last_tick']):
                overlaps.append([a['source_path'],b['source_path']])
    if any(sha256_file(p)!=sha for p,sha in protected.items()): raise ValueError('protected artifact changed')
    output={'version':'DUKASCOPY_H5_FEATURE_COVERAGE_V1','readiness':'NOT_READY' if failures or overlaps or
        any(r['quarantined_continuity_segments'] or any(m['missing_m15_intervals'] for m in r['months'].values()) for r in records) else 'READY',
        'scope':'Feature pipeline only, per-source coverage; no execution/trade outcomes. Source overlaps are NOT merged.',
        'model_sha256':sha256_file(DEFAULT_MODEL),'contract_checksum':CONTRACT_SHA256,'contract_file_sha256':contract_file_sha,
        'fit_calls':guard['fit_calls'],'continuity':evidence,'sources':records,'failures':failures,'identical_file_aliases':aliases,
        'ambiguous_overlapping_sources':overlaps,'detected_months':sorted(set(m for r in records for m in r['months'])),
        'protected_unchanged':True,'feature_order':list(FEATURES),
        'limitations':['Raw source files preserve all Bid/Ask and original duplicate order. Only derived BID candles are aggregated.',
            'No pre-2025 file selection is based on 2026 diagnostic outcomes.',
            'No new partition is certified or registered; this artifact cannot authorize a backtest or bypass PartitionService.',
            'Missing intervals are observed grid gaps, not evidence that ticks existed during closed markets.',
            'Filename/directory years are not trusted; monthly coverage uses actual UTC timestamps.',
            'Source-file edges are conservative partial boundaries. Cross-file history is not guessed or combined across ambiguous sources.']}
    publish(OUTPUT, output)
    print(json.dumps({'readiness':output['readiness'],'sources':len(records),'failures':len(failures),'months':len(output['detected_months'])}),flush=True)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',nargs='+',type=Path)
    run(parser.parse_args().roots)
