from __future__ import annotations
from .models import SnapshotLink,SnapshotPhase


def snapshot_signals(engine,frames,signal_ledger,*,symbol,base_timeframe="M15",dataset_id=None,partition_id=None):
    """Observe an existing signal ledger; feature values never flow back into strategy decisions."""
    snapshots=[];links=[]
    for signal in signal_ledger.records:
        snapshot=engine.snapshot(frames,symbol=symbol,decision_timestamp=signal.decision_timestamp,
            information_cutoff_timestamp=signal.information_cutoff,base_timeframe=base_timeframe,
            phase=SnapshotPhase.ENTRY_DECISION,dataset_id=dataset_id,partition_id=partition_id)
        snapshots.append(snapshot);links.append(SnapshotLink(snapshot.snapshot_id,SnapshotPhase.ENTRY_DECISION,signal.setup_id,signal.signal_id,signal.signal_id))
    return tuple(snapshots),tuple(links)
