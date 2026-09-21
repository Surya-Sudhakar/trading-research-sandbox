"""Read-only January H5 upstream warm-up investigation; never repairs or imputes."""
from datetime import timedelta
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from sandbox.market_state.universal.engine import UniversalFeatureEngine
from sandbox.market_state.universal.continuity import QualityContext
from sandbox.market_state.universal.multitimeframe import completed_bars
from sandbox.market_state.universal.volatility import atr_series
from sandbox.market_state.universal.market_metrics import atr_change
from sandbox.partition.service import PartitionService
from sandbox.partition.models import AccessContext, AccessOperation
from sandbox.research.registry import ResearchRegistry
from sandbox.provenance import sha256_file
from . import h5_january as h5


def trace_entry(source, entry):
    """Only completed source bars within the entry's recorded segment can contribute."""
    entry = pd.Timestamp(entry)
    if entry.tzinfo is None or not h5.START <= entry < h5.END:
        raise ValueError('aware January 2016 entry required')
    decision = entry.floor('15min')
    past = source.loc[source.timestamp_utc + timedelta(minutes=15) <= decision].copy()
    if past.empty or past.timestamp_utc.iloc[-1] + timedelta(minutes=15) != decision:
        raise ValueError('entry feature boundary missing from source')
    segment = past.continuity_segment_id.iloc[-1]
    current = past.loc[past.continuity_segment_id.eq(segment)]
    if not current.timestamp_utc.diff().iloc[1:].eq(timedelta(minutes=15)).all():
        raise ValueError('unrecorded gap inside source segment')
    bars = completed_bars(current, 'M15', ('H3',))['H3']
    atr = atr_series(bars.high.tolist(), bars.low.tolist(), bars.close.tolist())
    i = len(bars)-1
    numerator = atr[i] if i >= 0 else None
    denominator = atr[i-4] if i >= 4 else None
    calculated = atr_change(atr, i, 4) if i >= 0 else None
    first_full_open = current.timestamp_utc.iloc[0].ceil('3h')
    inputs = []
    for j, b in enumerate(bars.itertuples()):
        inputs.append({'open_time': b.timestamp_utc.isoformat(),
                       'close_time': (b.timestamp_utc+timedelta(hours=3)).isoformat(),
                       'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
                       'atr_14': atr[j]})
    previous = past.loc[past.timestamp_utc < current.timestamp_utc.iloc[0]]
    return {'entry_time': entry.isoformat(), 'feature_timestamp': decision.isoformat(),
        'segment_id': int(segment), 'segment_start': current.timestamp_utc.iloc[0].isoformat(),
        'previous_segment_last_open': previous.timestamp_utc.iloc[-1].isoformat() if len(previous) else None,
        'source_m15_bars_available': len(current), 'completed_h3_bars': len(bars),
        'h3_bars_required': 19, 'h3_bar_shortfall': max(0, 19-len(bars)),
        'latest_h3_close': inputs[-1]['close_time'] if inputs else None,
        'atr_current': numerator, 'atr_four_h3_bars_ago': denominator,
        'calculated_h3_atr_change_4': calculated,
        'reason': ('NO_COMPLETE_H3' if i < 0 else 'ATR_CURRENT_AND_LAG_NOT_SEEDED' if numerator is None
                   else 'LAGGED_ATR_NOT_SEEDED' if denominator is None else 'READY'),
        'first_possible_ready_time_if_segment_continues': (first_full_open+timedelta(hours=57)).isoformat(),
        'earliest_required_h3_open_for_this_entry': (decision.floor('3h')-timedelta(hours=57)).isoformat(),
        'h3_inputs': inputs}


def run():
    output = h5.OUTPUT/'h5_january_2016_warmup_diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    protected = set(h5.OUTPUT.glob('*.json')) | {h5.DEFAULT_MODEL, h5.EA,
        h5.OUTPUT/'h5_january_2016_trades.csv', Path('sandbox/test.xlsx')}
    before = {str(p): sha256_file(p) for p in sorted(protected)}
    with h5.no_fitting() as guard:
        contract, _, _ = h5.load_frozen_inputs()
        if sha256_file(h5.EA) != h5.EA_SHA:
            raise ValueError('EA changed')
        features = h5.read_features(contract['feature_artifact']['path'], contract)
        feature_path = Path(contract['feature_artifact']['path'])
        meta_path = feature_path.with_name('metadata.json')
        if sha256_file(meta_path) != contract['feature_artifact']['metadata_sha256']:
            raise ValueError('feature metadata changed')
        meta = json.loads(meta_path.read_text())
        lineage = [x for x in meta['transformation_history'] if x['step'] == 'AUTHORIZED_DISCOVERY']
        if len(lineage) != 1 or meta['quality_mode'] != 'SEGMENT_LOCAL_V1':
            raise ValueError('unexpected feature lineage')
        service = PartitionService(ResearchRegistry(Path('data/catalog.sqlite3')))
        manifest = service.get_partition(lineage[0]['partition_id'])
        if (manifest.source_dataset_id != meta['source_dataset_id'] or
                manifest.partition_fingerprint != meta['source_partition_fingerprint']):
            raise ValueError('partition provenance mismatch')
        source = service.access(manifest.partition_id, AccessContext.DISCOVERY_CONTEXT,
            AccessOperation.FEATURE_ANALYSIS, actor='h5-upstream-warmup-diagnostic',
            start_timestamp=h5.START, end_timestamp=h5.END)
        # Existing engine, existing segment semantics; no writes to the frozen artifact.
        replay = pd.DataFrame([r.model_dump() for r in UniversalFeatureEngine().compute(
            source, symbol='EURUSD', quality_context=QualityContext())])
        replay = replay.loc[replay.timestamp < h5.END].reset_index(drop=True)
        if not features.timestamp.reset_index(drop=True).equals(replay.timestamp):
            raise ValueError('replayed feature timestamps differ')
        left = features.loc[:, h5.FEATURES].to_numpy(float)
        right = replay.loc[:, h5.FEATURES].to_numpy(float)
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError('saved/replayed feature discrepancy requires investigation')
        trades = pd.read_csv(h5.OUTPUT/'h5_january_2016_trades.csv')
        unmatched = trades.loc[trades.state_at_entry.isna()]
        if len(unmatched) != 19:
            raise ValueError('unexpected H5 unmatched population')
        with sqlite3.connect('file:data/catalog.sqlite3?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            earlier = [dict(r) for r in db.execute("SELECT dataset_id,broker,symbol,timeframe,start,path FROM datasets WHERE symbol LIKE 'EURUSD%' AND start < '2016-01-01'")]
            line = db.execute('SELECT manifest_json FROM market_data_lineage WHERE dataset_id=?',
                              (meta['source_dataset_id'],)).fetchone()
        source_meta = json.loads(line[0])
        gap_path = Path(source_meta['artifact_directory'])/'gaps.parquet'
        gaps = pd.read_parquet(gap_path, filters=[('next_timestamp', '>=', '2016-01-01'), ('next_timestamp', '<', '2016-02-01')])
        gaps_by_segment = {int(x['next_segment_id']): x for x in gaps.to_dict('records')}
        audit = pd.read_parquet(Path(source_meta['artifact_directory'])/'audit.parquet',
                               columns=['timestamp_utc', 'classification', 'retained'],
                               filters=[('timestamp_utc', '>=', h5.START), ('timestamp_utc', '<', h5.END)])
        details = []
        for row in unmatched.itertuples():
            trace = trace_entry(source, row.entry_time)
            saved = features.loc[features.timestamp.eq(pd.Timestamp(trace['feature_timestamp']))].iloc[0]
            if not pd.isna(saved.h3_atr_change_4) or trace['calculated_h3_atr_change_4'] is not None:
                raise ValueError('null not reproduced')
            prefix = source.loc[(source.continuity_segment_id == trace['segment_id']) &
                                (source.timestamp_utc+timedelta(minutes=15) <= pd.Timestamp(trace['feature_timestamp']))]
            causal = UniversalFeatureEngine().compute(prefix, symbol='EURUSD')[-1]
            for name in h5.FEATURES:
                value = getattr(causal, name)
                if not ((pd.isna(saved[name]) and value is None) or saved[name] == value):
                    raise ValueError('causal-prefix replay mismatch')
            trace.update({'saved_h3_atr_change_4': None, 'prefix_replay_matches_all_14': True,
                          'assessment': 'LEGITIMATE_NULL_UNDER_FROZEN_SEGMENT_LOCAL_SEMANTICS',
                          'primary_cause': 'insufficient causal warm-up',
                          'contributing_cause': 'session/weekend gap handling' if trace['segment_id'] else 'available source history starts 2016-01-03',
                          'preceding_gap': gaps_by_segment.get(trace['segment_id'])})
            details.append(trace)
    after = {p: sha256_file(Path(p)) for p in before}
    if before != after:
        raise ValueError('protected artifact changed')
    report = {'experiment': 'H5_JANUARY_2016_UPSTREAM_WARMUP_DIAGNOSTIC_V1',
        'conclusion': 'No feature-engine or artifact-generation defect reproduced; do not change frozen segment semantics to remove legitimate nulls.',
        'feature_rows_replayed': len(replay), 'all_14_values_and_nulls_identical': True,
        'required_history': {'completed_h3_bars': 19, 'complete_m15_bars': 228,
            'continuous_hours_from_first_full_h3_open': 57, 'atr_seed_h3_bar_number': 15,
            'formula': 'ATR14[t]/ATR14[t-4]-1; 14 true ranges require 15 H3 bars, plus four lag bars.',
            'alignment': 'UTC 00/03/06/...; each H3 needs all 12 M15 bars. Partial buckets excluded.',
            'recursive_history': '19 is the minimum for a defined value; preserve ALL available prior bars in the same segment for the unchanged Wilder recurrence.'},
        'source_continuity_policy': source_meta['continuity_policy'], 'source_session_profile': source_meta['session_profile'],
        'earlier_registered_eurusd_datasets': earlier, 'earliest_registered_source': source_meta['first_timestamp'],
        'source_parent': source_meta['parent_raw_dataset_id'], 'source_timeframe': 'M15 (no M1 source in this lineage)',
        'january_source_rows': len(source), 'january_excluded_observations': int((~audit.retained).sum()),
        'january_gaps': gaps.to_dict('records'), 'affected_entries': details,
        'before_complete_coverage': 30, 'after_complete_coverage': 30, 'total_trades': 49,
        'fix_applied': False, 'h5_rerun': False, 'h5_pipeline_status': 'FAILED_INCOMPLETE_STATE_COVERAGE',
        'fit_calls': guard['fit_calls'], 'protected_before': before, 'protected_after': after,
        'scientific_limits': ['Weekend-associated gaps are not certified market closures. The source uses an unknown provider-session profile.',
            'Some stored segments start Sunday 00:00 UTC. Their session/timezone provenance cannot be corrected by assuming a normal FX reopen.',
            'Pre-2016 data cannot cure later January resets without crossing existing boundaries; no such registered EURUSD history was found.',
            'Any proposed session/timestamp or continuity-policy revision changes scientific inputs and must not be silently applied to the frozen model.',
            'No future observations, imputation, H5 rule changes, GMM fitting, or strategy changes.']}
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'replayed_rows': len(replay), 'affected': len(details), 'fix_applied': False}))
    return report


if __name__ == '__main__':
    run()
