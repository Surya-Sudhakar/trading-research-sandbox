# Multi-source research data V1

This package separates historical research storage from broker feeds. It adds
deterministic sanitation and continuity metadata without changing strategy or
backtester semantics.

## Source and artifact boundaries

- ExternalCSVAdapter interprets explicitly declared source timestamps and
  preserves original values for audit. MT5SourceAdapter uses the existing
  read-only MT5 rates API. ProviderAdapter is the source boundary.
- sanitize() maps these inputs into the common CanonicalMarketBar fields.
  Features remain provider-neutral and require only ordered completed OHLC.
- Raw external files and their existing QUARANTINED status are immutable.
  A SANITIZED derivative receives a new identity, source checksum, manifest,
  exclusion audit, gap inventory, and its own registered bar dataset.
- market_data_lineage stores SANITIZED-to-RAW parent links and sealed manifests.
  DatasetKind defines RAW, SANITIZED, RESAMPLED and FEATURE terminology.
  Resampling steps and FEATURE source/partition links are recorded in manifests.
- The existing Catalog/Provenance/PartitionService retain ownership of
  dataset verification and Discovery/Validation/Final-Test access.
- Feature Parquet files are content-addressed under data/derived/features.
  Sanitized bars live under data/derived/sanitized. Broker archive revisions
  live under data/broker_archive.

The canonical columns are timestamp_utc, symbol, provider, provider_symbol,
timeframe, price_type, open/high/low/close, nullable tick_volume/real_volume/
spread, source_dataset_id, quality_status, continuity_segment_id.
In sanitized bars source_dataset_id identifies the raw origin; the derivative
identity and parent link are in its manifest. Feature rows identify the
sanitized dataset actually used. Unknown optional values remain null in
storage. Only the compatibility *validation view* supplies neutral zeroes
to the old mandatory-field validator; those zeroes are not published prices,
volumes or spreads.

## MARKET_DATA_SANITATION_V1

Primary classifications are VALID, INVALID_TIMESTAMP, NONFINITE_PRICE,
NONPOSITIVE_PRICE, INVALID_OHLC, OFF_GRID, OTHER_INVALID,
DUPLICATE_IDENTICAL and DUPLICATE_CONFLICT.

Retain only VALID observations. Invalid observations are excluded, never
corrected. At a timestamp with identical OHLC and optional fields, retain
the first source observation deterministically and audit later identical
observations as DUPLICATE_IDENTICAL. At a conflicting timestamp, exclude
all observations, including a valid-looking one. Do not guess which is right.
Retained observations are sorted stably by UTC time.

Every source observation has a row number/classification/retained flag.
Excluded observations also retain original field strings. Raw source bytes
remain independently preserved and checksummed. No price interpolation,
forward fill, grid rounding or missing-bar creation occurs.

If an invalid timestamp cannot be located, it is excluded and every retained
bar receives a separate continuity segment. This fail-closed case publishes
no ready indicator history until the source interpretation is resolved.

Logical hashing covers ordered column names and canonical JSONL values,
normalizing nulls and UTC datetimes. Generation timestamps and Parquet
writer details are excluded from logical hashes. The dataset identity also
binds source identity, policy and session/continuity configuration.
The database registration is transactional and filesystem publication is
atomic. A complete unregistered artifact can be recovered after interruption,
with all hashes checked.

## Gaps and CONTINUITY_V1

Missing observations are not malformed prices. The inventory includes previous
and next timestamps, duration, missing slots, provider/symbol/timeframe,
classification, invalid-exclusion association, and reset decision.

SessionProfile requires timezone and documented evidence before weekly open/
close boundaries are accepted as expected. Wall-clock boundaries use IANA
timezone/DST rules. Exact documented weekly boundaries may yield
EXPECTED_WEEKEND. Friday-to-Sunday/Monday gaps not proven by a profile yield
WEEKEND_ASSOCIATED_UNKNOWN. Other gaps are UNEXPLAINED_GAP. Holiday/provider
session expectations remain unknown unless a future documented policy adds
them; no holiday is inferred automatically.

Defaults:

- reset when missing duration is at least 60 minutes;
- reset after any located invalid observation exclusion, even one missing bar;
- reset weekend-associated unknown gaps meeting the same threshold;
- allow history across an explicitly proven EXPECTED_WEEKEND;
- optionally reset even proven weekends with continue_expected_weekend=False.

Every reset starts new ATR, rank, momentum, structure and context histories.
H1/H3 reconstruction uses only the current segment, so partial aggregates
around excluded candles cannot become visible.

Use the optional feature mode explicitly:

```python
from sandbox.market_state.universal import UniversalFeatureEngine
from sandbox.market_state.universal.continuity import QualityContext

rows = UniversalFeatureEngine().compute(
    canonical_bars, symbol="EURUSD", quality_context=QualityContext()
)
```

Without quality_context, Step 1's original observed-bar behavior is unchanged.
The Discovery exporter automatically uses segment mode when segment metadata
exists. Complete ordinary datasets are regression-tested for identical output.
Current H3 remains UTC_FIXED. NEW_YORK_SESSION/CUSTOM_OFFSET anchors are
future work.

## Commands

Run from the existing repository:

```powershell
python -m sandbox.market_data sanitize RAW_DATASET_ID
python -m sandbox data partition create SANITIZED_ID PROGRAM_ID DISCOVERY EURUSD M15 START_UTC END_UTC
python -m sandbox.market_state.universal.dataset DISCOVERY_PARTITION_ID
python -m sandbox.market_data archive-once --symbol EURUSD --initial-days 7 --overlap-minutes 60
```

Sanitation never certifies the raw parent. Feature export accepts a partition
ID, never a free-form CSV path. The example actual run reserves 2025 onward
for Validation and exposes only the earlier source interval to Discovery.
Quality inspection is not strategy validation; no Final-Test claim is made.

## Broker archive

MT5 is a feed. poll() retrieves only from the last archived timestamp minus
overlap (or initial_days for an empty archive). It excludes still-forming M1
bars, rejects conflicting overlap, deduplicates identical observations, checks
strict values/grid and gaps, and publishes immutable monthly revisions.
A small atomic current.json pointer identifies each month's current revision.
Old Parquet revisions are retained. OS file locks are released even if a
writer process exits unexpectedly. Repeating an identical ingest publishes
no new revision. Daily rescheduling can be configured externally; this task
does not start an unattended scheduler.

Historical rate retrieval can still fail at the terminal even when quotes
are live. The archive does not fabricate bars in that case. Old revisions
are immutable; retention/compaction is intentionally not automated, so frequent
polling may grow storage over time.

## Provider comparison

compare_providers(a,b,pip_size=...) aligns aware unique timestamps. It reports
signed, absolute and pip differences for OHLC, direction agreement, range
difference/correlation and missing-on-A/B counts. No-match statistics are null.
Inputs should come from authorized Discovery access or a properly authorized
validation workflow. Matching prices are not required.

## Tests

```powershell
python -m pytest tests/test_multisource_data.py tests/test_real_multisource_data.py tests/test_universal_features.py -p no:cacheprovider
python -m pytest --ignore=tests/runtime -p no:cacheprovider
```

The real-data integration module runs when the actual sanitized Discovery
dataset is provisioned; otherwise it explicitly skips. Generated tests/runtime
files are not test source. API contract tests use an isolated one-import fixture
instead of assuming the user's live catalog always contains one import and
no research partitions.

