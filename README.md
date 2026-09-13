# Trading Research Sandbox — Stages 1–7

A new, isolated, read-only IC Markets MetaTrader 5 market-data foundation. It discovers broker symbols, inspects and retrieves M1 history, preserves immutable monthly Parquet files, audits integrity, and catalogs provenance in SQLite.

## Local retro-terminal UI and safe read API

The React/TypeScript frontend is in `frontend/`. It is a view/control surface over the authoritative Python implementation, never a second research engine. The isolated FastAPI adapter in `sandbox/api/` exposes explicit browser-safe DTOs at:

- `GET /api/system/status`
- `GET /api/strategies` and `/api/strategies/{strategy_id}`
- `GET /api/research/status`
- `GET /api/data/status`
- `POST /api/data/import` (controlled multipart upload and Stage 1 certification)
- `GET /api/evidence/status`
- `GET /api/audit/status`
- `GET /api/market-state/status`

Start local development in two terminals:

```powershell
# Terminal 1, from the project root
python -m sandbox api serve

# Terminal 2
cd frontend
npm install
npm run dev
```

The API binds to `127.0.0.1:8765` by default (port 8000 is reserved on the verified Windows host). CORS permits only `http://127.0.0.1:5173` and `http://localhost:5173`; the Vite server binds to `127.0.0.1:5173`. Set `VITE_SANDBOX_API_URL` only when intentionally using a different local API address. Build production assets with `cd frontend; npm run build`.

React cannot access SQLite, Parquet, MT5, paths, the research journal, access ledger, protected partitions, or the Final-Test vault directly. The sole controlled mutation route is historical upload: it streams CSV, CSV.GZ, or Parquet into a server-selected temporary directory, enforces `SANDBOX_UPLOAD_MAX_BYTES` (2 GiB by default), rejects unsafe filenames/extensions and invalid metadata, and delegates all parsing, validation, preservation, quarantine, duplicate detection, provenance, and certification to the existing `ExternalImportGateway`. Temporary upload files are removed after the gateway finishes. The browser cannot choose a destination.

The import UI requires explicit provider, symbol, TICK/M1 type, price representation, IANA source timezone, and either `ENGINEERING_DIAGNOSTIC` or `UNASSIGNED_RESEARCH_DATA`. Validation and Final-Test classifications are not accepted. A quarantined dataset has no certification bypass. Importing never creates a partition, hypothesis, experiment, candidate, execution run, or broker order. All other mutation, shell, strategy-loading, broker-order, Validation-data, and Final-data routes remain absent. Detailed failures remain server-side and browser errors contain no paths or credentials. Stage 6 authorization remains inside the existing sealed services. Catalog datasets are projected without internal paths.

Default integrated mode always uses live backend responses. When the backend is unavailable, the UI displays `BACKEND OFFLINE` and never substitutes mock results. No mock mode is shipped. Market-state status reports the installed causal framework but does not fabricate snapshots; Stage 7 evidence remains empty until legitimate reports exist. Known limitation: this first integration is deliberately read-only and does not yet serve the built frontend from Python or provide authenticated remote deployment—the intended deployment is localhost only.

## External historical-data import (Stage 1 extension)

Local CSV, CSV.GZ, and Parquet files can enter the Stage 1 trust boundary without replacing the existing MT5 download path. Imports support M1 OHLC or BID/ASK ticks, require an explicit provider, symbol, price type, timezone, and exact column mapping, preserve byte-identical source components under `data/external_raw`, and create canonical artifacts under `data/derived/external`. No downloader or network client is included.

An import is two-phase. `import-file` hashes and preserves every component, validates schema, timestamps, prices, ordering, gaps, overlaps, and quote relationships, then leaves clean data `STAGED`; material faults are `QUARANTINED`. `certify` rechecks both the caller's original component and preserved copy plus the canonical checksum before atomically adding the dataset to the Stage 1 catalog. Certification assigns `UNASSIGNED_RESEARCH_DATA`; it never creates a Discovery, Validation, or Final-Test partition.

```powershell
python -m sandbox data import-file --file C:\data\EURUSD_ticks.csv --provider DUKASCOPY --symbol EURUSD --resolution TICK --price-type BID_ASK --timezone UTC --derive-mid
python -m sandbox data import-file --file C:\data\EURUSD_M1.csv --provider DUKASCOPY --symbol EURUSD --resolution M1 --price-type BID --timezone UTC
python -m sandbox data import-list
python -m sandbox data import-inspect IMP-ID
python -m sandbox data import-status IMP-ID
python -m sandbox data certify EXT-DATASET-ID
```

Use repeated `--file` arguments for monthly or yearly components. Component order is canonicalized by resolved path, and each component retains its own SHA-256. Exact file plus interpretation duplicates return `DUPLICATE_DATASET`; a different interpretation of identical bytes requires `--interpretation-reason` and is flagged `MULTIPLE_INTERPRETATIONS`. Provider identity is declared provenance, not cryptographic source authentication. `data_source_provider` and `target_execution_broker` remain separate.

Tick-to-M1 uses UTC half-open minute buckets. BID and ASK are derived separately; MID is available only if explicitly requested during import. Empty minutes remain absent. BID/ASK imports expose spread open/close/min/max/mean in `PRICE_DISTANCE`, never implicitly in pips. Import parsing, validation, gap scanning, and canonical Parquet writing are bounded-memory and batch-oriented. The convenience `derive_m1` return API materializes its substantially smaller M1 result in memory; source ticks are retained in Parquet and can be partitioned into monthly components.

## Generic strategy plugin interface

Strategies are deterministic components behind `sandbox.strategies`; no strategy-specific market logic belongs in the sandbox core. A plugin declares immutable metadata, symbols, causal timeframes, required fields, warm-up requirements, directions, capabilities, parameter schema, analysis timezone, signal schema, and explicit versions. The registry validates this contract and hashes the implementation source independently of the human-readable version. Scientifically material parameters, code, metadata, or timeframe requirements therefore change the combined strategy fingerprint.

The trusted runtime gives plugins a read-only `StrategyContext` containing immutable candle snapshots truncated to `information_cutoff <= decision_timestamp`. It exposes no DataFrame, dataset path, database connection, Stage 7 result, or partition lookup. Warm-up decisions are skipped. Multi-timeframe inputs must be supplied by controlled sandbox derivation; sealed Stage 6 helpers construct complete causal bars inside the authorized partition boundary. Validation and Final strategy runs check the frozen candidate's code and parameter fingerprints. Final signal generation and Stage 2 execution occur inside the sealed service and do not expose Final candles to callers.

Plugins emit standardized signals with separate `setup_id` and `signal_id`. Geometry, timestamps, directions, and entry support are checked before conversion to Stage 2 `TradeIntent`; only `MARKET_NEXT_OPEN` and `MARKET_CLOSE` are currently accepted. The immutable Parquet signal ledger records code, parameter, and per-signal fingerprints, preserving the chain from data to signal to Stage 2 ledger. Running the same plugin twice is used to detect differing signal-ledger fingerprints and hidden nondeterminism.

```powershell
python -m sandbox strategies list
python -m sandbox strategies inspect SYNTHETIC_INTERFACE_TEST
python -m sandbox strategies validate SYNTHETIC_INTERFACE_TEST
python -m sandbox strategies fingerprint SYNTHETIC_INTERFACE_TEST
```

`SYNTHETIC_INTERFACE_TEST` is deliberately mechanical and exists only as an interface fixture. It is marked `synthetic_test_plugin` and is not registered as a real research strategy or family. Python plugins execute in-process: the API provides scientific isolation and causal data minimization, not a security sandbox for hostile arbitrary Python. Untrusted code could use Python itself to access files or operating-system APIs and must not be installed. No optimizer, ranking command, strategy discovery, or live broker execution is included.

## S001 v1.0 — Intraday Trend Momentum Pullback Continuation

S001 is the first real strategy component, but it has only been implementation-tested with synthetic OHLC fixtures. **S001 v1.0 has not yet been shown to be profitable. Passing implementation tests is not evidence of a trading edge.** It has not been run against historical EURUSD research data and has not created a hypothesis, experiment, candidate, or research partition.

The exact baseline is:

- EURUSD, LONG and SHORT; completed H1 context and completed M15 setup bars.
- H1 trend: latest 12-bar close-to-oldest-open displacement divided by causal Wilder/RMA ATR(14), requiring 1.50 ATR in the direction.
- M15 momentum: latest 8-bar equivalent displacement divided by Wilder/RMA ATR(14), requiring 1.25 ATR aligned with H1.
- ATR true range is `max(high-low, abs(high-previous_close), abs(low-previous_close))`. The first ATR is the arithmetic mean of 14 TR values and later values use Wilder recurrence.
- Ordered impulse anchors use the earliest tied qualifying extreme that still has a strictly later opposite anchor.
- A subsequent lower low starts a bullish pullback; a subsequent higher high starts a bearish pullback. The impulse then remains frozen.
- Pullback below 25% waits, 25%–60% inclusive is armed, and above 60% invalidates before any same-bar confirmation. Recorded depths are not clamped; inclusive boundary comparisons use a fixed `1e-12` floating-point tolerance.
- Confirmation is a strict close beyond the immediately previous M15 high for LONG or low for SHORT. Wick penetration and equality do not qualify. Bars 1–8 may confirm; otherwise the setup expires before bar 9.
- H1 direction is causally recalculated at confirmation. Context loss invalidates the setup.
- Entry intent is `MARKET_CLOSE`. The stop is the complete pullback extreme through the confirmation bar; the target is fixed at exactly 2R. Stage 2 remains authoritative for fills, gaps, costs, same-bar outcomes, and results.
- At most one active setup per direction and one unresolved S001 position per symbol are allowed. After emission, the orchestrator must explicitly notify the plugin when Stage 2 resolves the position before another intent can be emitted.

Warm-up is 15 completed H1 and 15 completed M15 bars, accounting for ATR(14)'s previous close. Signal metadata records the causal H1/M15 measurements, frozen anchors, depth, duration, confirmation candle measurements, direction, and deterministic UTC/New York calendar values; calendar values are explanatory and never filters.

Baseline-assumption alternatives are documented in [the S001 component README](sandbox/strategies/s001/README.md). They are not a Cartesian sweep, have not been executed, and may only be considered later through separately registered Discovery experiments.

Stage 1 deliberately contains **no strategy, signal, backtest, optimizer, machine-learning, or trading/execution code**.

Stage 1 is frozen. Stage 2 adds a historical simulation engine only: it accepts externally supplied trade intents and never generates signals or communicates orders to MT5.

## Requirements and setup

- Windows with the IC Markets MT5 terminal installed and logged in
- Python 3.12+

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `SANDBOX_MT5_TERMINAL_PATH` only if automatic terminal discovery does not select the intended IC Markets terminal. No account password is required or logged; the adapter uses the terminal's existing session.

## Commands

```powershell
python -m sandbox mt5 status
python -m sandbox symbols search EURUSD
python -m sandbox history inspect EURUSD
python -m sandbox history download EURUSD --timeframe M1 --start 2024-01-01T00:00:00Z --end 2024-01-02T00:00:00Z
python -m sandbox data audit EURUSD.a
python -m sandbox data catalog
```

`history inspect` deliberately queries the terminal rather than assuming a fixed retention period. Its reported bar count is the count MT5 actually returned over the inspection window; MT5 terminal chart/history settings can constrain availability. Large inspections can take time.

## Storage and immutability

Raw files use:

```text
data/raw/ic_markets/<server>/<broker-symbol>/M1/YYYY/MM.parquet
```

Each file has a sidecar `MM.metadata.json`. A second write to an existing raw month fails instead of overwriting. Invalid datasets are written to `data/quarantine`. The SQLite catalog stores only metadata, never candles. A successful download rereads each Parquet file, calculates SHA-256 over its exact bytes, and records provenance.

Resumption is explicit and safe: request only missing time ranges. Existing months are refused, preventing an accidental mutation. For partial-month extension, download to a separate empty data directory or preserve/export the original month and perform a reviewed new ingestion; Stage 1 does not silently merge into an immutable raw file.

## Integrity meanings

- `expected_market_closure`: recognized Friday-to-Sunday FX closure; informational.
- `suspicious_gap`: a non-weekend gap of five minutes or more; needs investigation.
- `confirmed_data_error`: duplicates, ordering defects, invalid/null prices, schema defects, or short missing intervals.

No replacement candle is generated. Note that holidays, symbol-specific sessions, and broker DST transitions are not inferable from M1 candles alone and can remain warnings requiring human review.

## Tests

```powershell
pytest
```

Tests mock MT5 and require no live broker connection.

## Safety boundary

The MT5 adapter exposes only connection status, symbol discovery/resolution, and `copy_rates_range` historical reads. It has no order, position, pending-order, or account-operation methods. Keep the terminal configured according to your own operational security policy even though this code has no execution API.

## Stage 2 architecture

```text
Market data → external trade intent → execution engine → position state
            → exit resolution → immutable trade ledger → descriptive metrics
```

The engine contains no trend, momentum, indicator, session, pair, or strategy assumptions. Supported entry instructions are `MARKET_NEXT_OPEN` and `MARKET_CLOSE`. `LIMIT` and `STOP` are modeled values but deliberately rejected until exact semantics are implemented; they are never approximated.

### Timestamp and no-look-ahead contract

Market candle `timestamp_utc` is the bar-open timestamp. `ExecutionConfig.candle_interval_seconds` explicitly defines its duration (60 by default).

- `MARKET_NEXT_OPEN` uses the first candle whose open timestamp is at or after the signal-availability timestamp.
- `MARKET_CLOSE` uses a candle only when `bar_open + candle_interval == signal_timestamp`. Thus its close was available at the stated signal time.
- Signals outside the dataset, non-aligned close signals, and absent next-open candles are preserved as rejected ledger rows.

Input candles are normalized into stable UTC ascending order. Duplicate timestamps are rejected. Execution never reads a pre-signal candle to choose a fill.

### R multiples

For a long, `1R = entry_price - initial_stop_loss`; for a short, `1R = initial_stop_loss - entry_price`. Non-positive risk is rejected. Gross and net R use that original risk distance and support arbitrary valid targets.

### Exit and gap assumptions

Stage 1 MT5 OHLC is treated as bid data. Long barriers use bid OHLC. For shorts, ask OHLC is reconstructed as bid OHLC plus the configured spread distance. If SL and TP are both touched in one candle, `CONSERVATIVE` resolves to the stop with `SAME_BAR_AMBIGUOUS_SL_FIRST`. OHLC cannot reveal the true intrabar path; lower-timeframe disambiguation is intentionally left as a future extension.

With `CONSERVATIVE_OPEN`, a candle reopening beyond a stop exits at its adverse open, not the stop price. A favorable gap beyond a target receives only the target price. Missing intervals and weekend reopenings are counted in each ledger record and emitted to audit hooks. Candles are never synthesized.

### Costs

Costs are deterministic and separate from gross P&L:

- Spread: `ZERO_SPREAD`, `FIXED_SPREAD`, or `HISTORICAL_CANDLE_SPREAD`. MT5 candle spread is an integer number of symbol **points**, converted as `spread × symbol_point`; it is not assumed to be pips.
- Commission: `ZERO_COMMISSION`, `FIXED_PER_TRADE`, or `FIXED_PER_LOT_PER_SIDE` (round trip = configured per-side amount × quantity × 2).
- Slippage: `ZERO_SLIPPAGE` or adverse `FIXED_PRICE_SLIPPAGE`. There is no randomness.

`entry_price`, `exit_price`, and `gross_pnl` are frictionless bid-reference simulation values. `spread_cost`, `commission_cost`, and `slippage_cost` are explicit deductions, and `net_pnl = gross_pnl - all costs`. Short barrier detection still uses reconstructed ask OHLC.

### Reproducibility and evidence

The SHA-256 run fingerprint hashes sorted dataset IDs, canonical trade intents, the complete serialized execution configuration, and engine/ledger versions. Identical inputs produce an identical ledger and fingerprint. Every intent produces a ledger row, including rejected and end-of-data positions. Ledgers are Parquet files under `results/ledgers`; run configuration and summaries are stored separately. SQLite `research_runs` is added with `CREATE TABLE IF NOT EXISTS`, preserving Stage 1 tables.

Metrics are descriptive and ledger-only: counts, win/loss rates, gross/net R, distribution, profit factor, R drawdown, streaks, holding duration and annualized frequency, plus yearly and symbol breakdowns. No optimization or ranking exists.

### Stage 2 CLI

Trade intents are a JSON array matching `TradeIntent`; configuration is a JSON object matching `ExecutionConfig`.

```powershell
python -m sandbox backtest run candles.parquet intents.json --config execution-config.json
python -m sandbox backtest inspect RUN_ID
python -m sandbox backtest ledger RUN_ID
python -m sandbox backtest summary RUN_ID
```

Omitting `--config` uses `ExecutionConfig`, whose complete defaults are still serialized into the run artifact and fingerprint.

### Stage 2 limitations

- Candle simulation cannot determine intrabar ordering.
- Historical spread is a candle snapshot/aggregate supplied by MT5, not tick-by-tick ask history.
- Fixed costs are expressed in the same normalized P&L units as price-distance × quantity; currency conversion and contract-value accounting are future work.
- Open positions at end of data remain explicit `OPEN` ledger rows and are not force-closed.
- LIMIT and STOP entries are unsupported and rejected.

## Stage 3 research registry

Stage 3 records the hierarchy `Program → Question → Hypothesis → Experiment → Backtest Run → Result → Verdict`. It does not interpret or generate strategy rules. Strategy and evaluation specifications are opaque canonical JSON documents.

An experiment begins as `DRAFT`. Approval requires an existing question, an approved hypothesis, at least one registered dataset with valid provenance and checksum, a strategy specification, a complete Stage 2 execution configuration, pre-registered evaluation criteria, a code version, and an experiment fingerprint. Approved specifications cannot be edited. Material changes require a child experiment with `parent_experiment_id` and `change_reason`.

Valid lifecycle paths are enforced:

```text
DRAFT → APPROVED → RUNNING → COMPLETED → REVIEWED → CLOSED
                    └──────→ FAILED
DRAFT/APPROVED/RUNNING → CANCELLED where permitted
```

Canonical SHA-256 experiment fingerprints include question, hypothesis, sorted dataset IDs, opaque strategy specification, complete execution configuration, evaluation criteria, code version, and research schema version. Filesystem locations and dictionary insertion order do not affect scientific identity. An identical approved fingerprint is reported as `DUPLICATE_EXPERIMENT`; an explicit replication links a new run to the original experiment.

Every meaningful registry event is appended to SQLite `research_journal`. Each entry hashes canonical entry content and the prior entry hash. `research journal verify` detects accidental modification of old history. Approved/completed scientific objects have no delete operations.

Stage 2 results are ingested from verified run artifacts, not manually entered. The registry checks engine version, execution configuration, dataset IDs, dataset checksums, and ledger checksum. Result metrics remain immutable; a verdict is a separate record linked to the exact result.

Completed research can be rendered under `results/experiments/EXP-.../manifest.json` and `manifest.md`. The manifest includes hierarchy, frozen specification, dataset provenance, code/execution versions, run fingerprints, metrics, checksums, verdicts, ancestry, and timestamps. The timeline command renders recorded relationships without scientific interpretation.

### Research CLI

Create commands accept JSON documents containing the corresponding model fields:

```powershell
python -m sandbox research program create program.json
python -m sandbox research question create question.json
python -m sandbox research hypothesis create hypothesis.json
python -m sandbox research hypothesis approve H001
python -m sandbox research experiment create experiment.json
python -m sandbox research experiment approve EXP-000001
python -m sandbox research experiment start EXP-000001
python -m sandbox research experiment link-run EXP-000001 run-...
python -m sandbox research experiment complete EXP-000001
python -m sandbox research experiment inspect EXP-000001
python -m sandbox research experiment ancestry EXP-000001
python -m sandbox research experiment replicate EXP-000001 RUN-ID RUN-FINGERPRINT
python -m sandbox research experiment manifest EXP-000001
python -m sandbox research verdict record EXP-000001 RES-ID FAIL --rationale "Pre-registered criterion not met" --decided-by researcher
python -m sandbox research timeline PROGRAM_ID
python -m sandbox research journal verify
```

When Git metadata is unavailable, code version capture hashes the `sandbox/**/*.py` source tree and records working-tree state as `UNKNOWN`. It never creates commits or modifies Git history.

### Stage 3 limitations

- No automatic scientific evaluation or diversion detection is performed.
- No authorization/identity system exists; `decided_by` is an explicit recorded label.
- The hash chain detects accidental modification but is not protection from a malicious database administrator.
- Discovery/Validation/Final dataset enforcement and robustness analysis remain future stages.

## Stage 4 deterministic research auditor

Stage 4 evaluates methodology, never profitability. PRE_RUN audits inspect the frozen specification, chronology, ancestry, family history, dataset usage, pre-registration, integrity, complexity and execution compatibility. They do not read win rate, net R, profit factor, drawdown, or individual winners/losers. POST_RUN audits may additionally label sample size and execution evidence without performing statistical robustness analysis.

Every audit is immutable and records auditor version `1.0.0`, the fingerprint of [research_audit.json](</C:/Users/surya/Documents/Codex/2026-08-27/files-pasted-by-the-user-codex/outputs/trading-research-sandbox/config/research_audit.json>), phase, findings, transparent score, and separate hard-block reasons. Repeating the same audit state/config/version returns the same deterministic audit record; changed semantics require a new auditor version.

Classifications:

- `SAFE`: no meaningful detected risk; execution may proceed.
- `WARNING`: execution may proceed with permanent findings.
- `HIGH_RISK`: execution requires a stored override with reason and approver.
- `BLOCKED`: execution is prohibited and cannot be overridden.

The score is capped at 100 and adds 10 points per WARNING and 25 per HIGH finding. INFO contributes zero. CRITICAL hard blocks remain separate and force `BLOCKED`. The score is a research-risk signal, not a probability that a strategy is false.

Hard blocks cover invalid dataset checksums, corrupted relevant journal history, approved-spec fingerprint mutation, incompatible execution configuration, absent approval, explicit/impossible information chronology, and known discovery/final-test contamination when independence is claimed.

Strategy temporal metadata uses an opaque `strategy_spec.temporal_contract` with `information_available_at`, `signal_timestamp`, `execution_timestamp`, `source_timeframe`, `requires_future_bars`, and `confirmation_delay`. Future-bar confirmation is acceptable only when signal/execution occur after information becomes available.

Dataset usage records `UNSEEN`, `DISCOVERY`, `VALIDATION`, `FINAL_TEST`, or `DIAGNOSTIC`. This is auditable metadata, not yet Stage 6 access locking. Result-driven descendants inherit their experiment family. Structural comparison reports changed field paths and repeated values without selecting or recommending a parameter.

Observable degrees of freedom, complexity, hypothesis counts, near duplicates and sample labels are preserved as transparent evidence. Default post-run labels are `<30 VERY_SMALL_SAMPLE`, `30–99 SMALL_SAMPLE`, and `>=100 BASIC_SAMPLE_THRESHOLD_MET`; none establish robustness.

Research execution through `experiment start` always runs PRE_RUN audit gating. Stage 2 remains independently callable for low-level synthetic testing. Linking a Stage 2 result creates a stored POST_RUN audit. PASS verdicts that conflict with supported frozen minimum criteria are rejected as `VERDICT_CRITERIA_MISMATCH`; metrics are never changed.

```powershell
python -m sandbox research audit pre EXP-000001
python -m sandbox research audit post EXP-000001
python -m sandbox research audit inspect AUDIT-ID
python -m sandbox research audit history EXP-000001
python -m sandbox research audit override AUDIT-ID --reason "Documented discovery continuation" --approved-by researcher
python -m sandbox research family inspect FAMILY-ID
python -m sandbox research family parameters FAMILY-ID
```

Audit JSON/Markdown reports are stored under `results/audits`. Experiment manifests include audits and overrides. If regenerated scientific context differs, a fingerprint-suffixed manifest is created instead of overwriting the historical manifest.

### Stage 4 limitations

- Temporal and falsifiability checks are deterministic structural checks, not natural-language understanding.
- Dataset usage is recorded but strict Discovery/Validation/Final access control is deferred.
- Risk thresholds are explicit heuristics, not statistical evidence.
- No best parameter, strategy rule, hypothesis, or performance recommendation is generated.

## Stage 5 question and decision control

Stage 5 makes navigation explicit while preserving discovery freedom. Every controlled question, hypothesis and experiment records `DISCOVERY`, `VALIDATION`, or `FINAL_TEST`; evidence origins record whether an idea came from theory, prior/external research, discovery/validation/final data, or a bug/data/execution investigation. Discovery branches and result-driven changes remain allowed when labeled and journaled, but their evidence cannot later be represented as independent validation.

Controlled non-root questions require a parent, purpose, relevance to the parent, evidence origins and mode. Controlled experiment approval additionally requires structured answers for the question answered, why it matters, motivating evidence/theory, what PASS and FAIL teach, and pre-registered outcome actions. Natural-language quality remains a manual-review responsibility.

Decision nodes preserve possible outcomes, mapped actions and optional stopping conditions. Recording a decision selects only a pre-registered outcome; the system never invents an action, hypothesis or strategy.

Candidate records are generic opaque strategy specifications. Freezing preserves the strategy, execution/evaluation configuration, symbols, timeframes, session rules, code version, discovery datasets, lineage and fingerprint. Frozen candidates cannot be edited. A material change creates `CAND-...-v2`, returns to `EXPLORATORY`, and inherits no validation success.

Logical validation eligibility requires a frozen complete candidate, discovery lineage, Stage 4 audit history, registered validation data and no discovery usage of that data. Validation failure rejects that version and records validation exposure; modified descendants return to discovery. Stage 6 will add physical access locking.

Strategy families can be exhausted without closing Q0. Search-burden and candidate-card reports expose questions, hypotheses, experiments, parameter variants, result-driven changes, datasets and rejected related candidates without converting these counts into probabilities.

Navigation commands report the question tree, active registered decisions, approved work, open branches, dead ends and human-decision points:

```powershell
python -m sandbox research status PROGRAM_ID
python -m sandbox research tree PROGRAM_ID
python -m sandbox research next PROGRAM_ID
python -m sandbox research question inspect Q_ID
python -m sandbox research question close Q_ID --reason "..." --by researcher
python -m sandbox research question reopen Q_ID --reason "..." --new-evidence "..." --by researcher
python -m sandbox research branch inspect Q_ID
python -m sandbox research decision inspect DECISION_ID
python -m sandbox research candidate inspect CAND_ID
python -m sandbox research candidate freeze CAND_ID --validation-dataset DATASET_ID
python -m sandbox research candidate lineage CAND_ID
python -m sandbox research candidate burden CAND_ID
python -m sandbox research candidate card CAND_ID
python -m sandbox research family inspect FAMILY_ID
```

`research next` returns only unresolved registered work or `NO PRE-REGISTERED NEXT STEP. Human research decision required.` It never proposes indicators, rules, parameters or hypotheses.

### Stage 5 limitations

- Diversion classification uses explicit relationships and metadata, not semantic text analysis.
- Validation and final-test eligibility are logical metadata gates; hard partition locking is deferred to Stage 6.
- No candidate performance ranking or automatic stopping decision is performed.

## Stage 6 partition vault and access control

Stage 6 creates deterministic chronological `[start, end)` UTC partitions with explicit `DISCOVERY`, `VALIDATION`, `FINAL_TEST`, or `DIAGNOSTIC` roles. Roles are stored metadata and never inferred from filenames. Definition fingerprints hash the source checksum, boundaries, role, symbol, timeframe and policy version. Immutable manifests add the partition checksum, row count, schema and manifest fingerprint.

Scientific Discovery/Validation/Final partitions cannot overlap. Explicitly declared Diagnostic partitions may overlap but cannot be represented as independent evidence. Source Parquet remains unchanged; derived files live under `data/derived/partitions`, while Final partitions can use the separate `data/final_vault` root.

Every access declares `DISCOVERY_CONTEXT`, `VALIDATION_CONTEXT`, `FINAL_TEST_CONTEXT`, or `SYSTEM_INTEGRITY_CONTEXT` plus an operation. Discovery can read only Discovery or declared Diagnostic values. Validation requires a frozen candidate and immutable candidate/partition/configuration/evaluation binding. Generic Final-Test reads, statistics, analysis and exports are denied even after authorization; final data is released only inside sealed execution.

```text
successful validation → explicit authorization → fingerprint verification
→ sealed Stage 2 run → immutable ledger/result → permanent FINAL_TEST_EXPOSURE
```

Repeating the same candidate/partition relationship as an independent final test is denied. Failed results remain stored. Modified candidates return to Discovery and inherit no validation or final evidence.

Every allowed or denied access is appended to a hash-chained ledger with actor, context, operation, decision, reason, rows returned and access-policy version. Contamination is scoped by program, candidate and candidate-base lineage. Unauthorized final access is CRITICAL evidence visible to Stage 4.

`SYSTEM_INTEGRITY_CONTEXT` returns only file existence, checksum, schema and row-count validity. It cannot return candles or market aggregates. Sanitized metadata contains boundaries, role, row count, checksum and fingerprint, but no price or performance statistics.

Derived higher-timeframe candles use only complete M1 groups wholly contained in one partition. Validation warm-up returns a marked window with `trading_allowed=false` and `included_in_evaluation=false`, logged separately as `WARMUP_READ`.

```powershell
python -m sandbox data partition create DATASET PROGRAM ROLE SYMBOL M1 START END
python -m sandbox data partition inspect PARTITION_ID
python -m sandbox data partition verify PARTITION_ID
python -m sandbox data partition access-history PARTITION_ID
python -m sandbox research contamination inspect CANDIDATE_ID
python -m sandbox research validation bind CANDIDATE_ID PARTITION_ID
python -m sandbox research validation run CANDIDATE_ID PARTITION_ID intents.json
python -m sandbox research final authorize CANDIDATE_ID PARTITION_ID VALIDATION_RESULT --authorized-by researcher
python -m sandbox research final run CANDIDATE_ID PARTITION_ID intents.json
```

Versions: `partition_schema=1.0.0`, `partition_policy=1.0.0`, `access_policy=1.0.0`.

### Stage 6 security boundary

This is application-level research isolation, access auditing and contamination detection—not adversarial security. A Windows administrator can still copy or open Parquet files outside the application. Generic sandbox CLI/API paths enforce the vault policy and make research-process access permanent and reviewable.

The short May–August 2026 IC Markets dataset is engineering/diagnostic material only. It has not been divided into real research partitions and must not be treated as untouched future evidence.

## Stage 7 statistical evidence

Stage 7 derives immutable, fingerprinted evidence reports from verified Stage 2 Parquet ledgers. It separates observed performance from uncertainty, robustness, and independent validation. Reports cover sample composition, realized R, Wilson win-rate intervals, deterministic IID and moving-block bootstrap intervals, dependence, yearly/frequency stability, partial years, symbols, directions, concentration, drawdown, historical-outcome resampling, order sensitivity, and cross-symbol overlap.

The versioned defaults are in `config/statistical_evidence.json`; the evidence engine version is `1.0.0`. Fingerprints bind the ledger checksum, configuration, engine version, mode, methods, seeds, profiles, and warnings. A ledger checksum mismatch blocks analysis. Wilson intervals have a frequentist repeated-sampling meaning. Bootstrap and permutation outputs are stress tests—not posterior probabilities, future-market predictions, or a probabilistic market model.

Sensitivity accepts registered variants only and never generates values or recommends a best parameter. Cost analysis accepts registered scenarios and preserves the declared baseline. Discovery search burden and subgroup multiplicity remain visible. Warnings describe evidence and do not override the frozen EvaluationSpec.

```powershell
python -m sandbox stats analyze RUN_ID --mode DISCOVERY
python -m sandbox stats inspect REPORT_ID
python -m sandbox stats yearly REPORT_ID
python -m sandbox stats symbols REPORT_ID
python -m sandbox stats directions REPORT_ID
python -m sandbox stats uncertainty REPORT_ID
python -m sandbox stats drawdown REPORT_ID
python -m sandbox stats sensitivity registered-variants.json threshold
python -m sandbox stats costs registered-cost-scenarios.json
```

Stage 6 remains authoritative: Discovery cannot analyze Validation or Final-Test partitions, Validation requires its frozen binding, and generic Final-Test analysis is denied. Final statistics must execute within the pre-authorized sealed path. Reports are included in experiment manifests and referenced by the tamper-evident journal.

Limitations: bootstrap results depend on exchangeability or the declared block length; no risk-of-ruin claim, market-regime discovery, feature mining, parameter generation, optimization, or strategy generation is included. The current IC Markets sample remains `ENGINEERING_DIAGNOSTIC` and is not strategy evidence.
# Generic pre-entry market-state framework

`sandbox.market_state` is a strategy-independent, deterministic measurement layer. It accepts already-authorized, boundary-safe candle frames and an explicit information cutoff. A candle is available only when its full timeframe has closed at or before that cutoff. The engine cannot read partitions, Stage 7 results, trade outcomes, or candidate verdicts, and none of its measurements can create, suppress, or alter a strategy signal.

The `MARKET_STATE_V1` schema records continuous trend, maturity, volatility, momentum, range, causally confirmed swing/structure, calendar, and clock-block measurements for configured timeframes and horizons. Defaults are M15/H1 horizons 4, 8, 12, and 24; Wilder ATR 14; short/long volatility windows 14/50; an inclusive weak ATR percentile rank over up to 250 trailing causal observations; and strict confirmed pivots with two left and two right bars. A pivot becomes available only after both right-side bars close. These defaults are measurement conventions, not optimized strategy parameters or eligibility thresholds.

Each immutable snapshot contains deterministic code/configuration fingerprints, validity and missing-feature maps, data-quality flags, and a deterministic ID derived from visible data and scientific configuration. Undefined denominators and inadequate history remain unavailable; values are never forward-filled. Snapshot ledgers persist measurements and separate setup/signal/trade links as immutable Parquet evidence. S001 linkage is performed externally after its signal ledger exists, so S001 v1.0 logic and identity remain unchanged.

OHLC data cannot reveal intrabar path, and this layer does not attempt to infer one. It also provides no session labels, continuation/exhaustion classifier, feature ranking, optimization, machine learning, or profitability analysis.
