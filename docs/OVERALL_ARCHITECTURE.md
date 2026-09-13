# Trading Research Sandbox — Overall Architecture

## Purpose

Trading Research Sandbox is an evidence-first research system for discovering, testing, rejecting, validating, and eventually exporting trading strategies without allowing the research process to silently contaminate protected data or rewrite failed history.

The intended end state is an **Autonomous Trading R&D System**. A human supplies a broad research question or hypothesis family; Sandbox constructs and evaluates candidate explanations and strategy variants, remembers both successes and failures, promotes only robust survivors, and keeps Validation and Final-Test data outside the discovery feedback loop.

> Implementation tests prove software behavior, not trading profitability. A candidate is not a supported strategy merely because code or CI passes.

## Architecture at a glance

```text
HUMAN / RESEARCH QUESTION
          |
          v
+-----------------------------+
| 1. MARKET DATA FOUNDATION   |
| MT5 / external imports      |
| Parquet + catalog           |
| hashes + provenance         |
+-------------+---------------+
              |
              v
+-----------------------------+
| 2. MARKET-STATE / FEATURES  |
| ATR / volatility            |
| sessions / geometry         |
| previous-day levels         |
| configurable pivots         |
| HH HL LH LL                 |
| BOS / CHoCH                 |
| sweep / reclaim             |
+-------------+---------------+
              |
              v
+-----------------------------+
| 3. CAUSAL DISCOVERY DATASET |
| predictors available at t   |
| research anchors            |
| future targets separated    |
| no look-ahead               |
+-------------+---------------+
              |
              v
+-----------------------------+
| 4. RESEARCH PIPELINE V1     |  IMPLEMENTED
| verify partition            |
| build features              |
| event study                 |
| metrics                     |
| immutable result package    |
+-------------+---------------+
              |
              v
+-----------------------------+
| 5. RESEARCH MEMORY V1       |  IMPLEMENTED
| DuckDB scientific memory    |
| run/candidate fingerprints  |
| parameters + metrics        |
| lineage                     |
| verdict + rejection reason  |
+-------------+---------------+
              |
              v
+-----------------------------+
| 6. AUTONOMOUS SEARCH        |  NEXT
| generate candidates         |
| vary conditions/parameters  |
| skip duplicates             |
| cheap screening             |
| rank/promote survivors      |
+-------------+---------------+
              |
              v
+-----------------------------+
| 7. EXECUTION / BACKTEST     |
| deterministic fills         |
| SL / TP / exits             |
| spread / commission         |
| slippage / sizing           |
| immutable trade ledger      |
+-------------+---------------+
              |
              v
+-----------------------------+
| 8. ROBUSTNESS SCREENING     |
| sample size                 |
| parameter sensitivity       |
| yearly/session stability    |
| cost sensitivity            |
| regime stability            |
+-------------+---------------+
              |
              v
+-----------------------------+
| 9. VALIDATION GUARDIAN      |
| Discovery                   |
|      -> Validation          |
|            -> Final-Test    |
| protected data boundaries   |
+-------------+---------------+
              |
              v
+-----------------------------+
| 10. RESEARCH DECISION       |
| REJECTED / WEAK             |
| PROMISING / SUPPORTED       |
| INCONCLUSIVE                |
+-------------+---------------+
              |
              +------------------------------+
              |                              |
              v                              |
+-----------------------------+              |
| 11. AUTONOMOUS RESEARCHER   |              |
| analyze failure patterns    |              |
| propose next investigation  |--------------+
| use research memory         |
+-------------+---------------+
              |
              v
+-----------------------------+
| 12. OPTIONAL ML             |
| LightGBM / ranking          |
| context/regime models       |
| only when OOS evidence helps|
+-------------+---------------+
              |
              v
+-----------------------------+
| 13. STRATEGY DSL            |
| machine-readable strategy   |
| frozen rules + parameters   |
+-------------+---------------+
              |
              v
+-----------------------------+
| 14. EXPORT / DEPLOYMENT     |
| Sandbox / Pine / MQL5       |
| StrategyTune-style UI       |
+-----------------------------+
```

## The three logical brains

### Market understanding

The data and market-state layers answer: **What was knowable at this point in time?** They create reusable causal primitives rather than embedding one strategy's assumptions into the core.

### Scientific research

The pipeline, event studies, deterministic backtester, robustness checks, protected partitions, and research memory answer: **Is there reproducible evidence for this behavior?**

### Autonomous researcher

The future autonomous-search/research loop answers: **Given what has already been tested, what is worth testing next?** It must use Discovery evidence and Research Memory without using Final-Test results to optimize a claim.

## System-of-record boundaries

These boundaries are intentional and should prevent the architecture from becoming several overlapping databases that disagree with each other.

| Component | Authoritative responsibility | Must not become |
|---|---|---|
| Parquet/raw archives | Market observations and immutable source artifacts | Research verdict database |
| Catalog / partition services | Dataset identity, provenance, partition role and authorization | Candidate-search memory |
| Research Registry + journal | Scientific governance: programs, questions, hypotheses, registered experiments, lifecycle, approvals and audit trail | High-volume candidate warehouse |
| Research Pipeline | Deterministic execution of one registered Discovery experiment | Autonomous hypothesis generator |
| Research Memory (DuckDB) | High-volume run/candidate search memory, lineage, metrics, failure reasons and duplicate detection | Second governance registry |
| Execution ledger | Fill/position/trade evidence | Hypothesis memory |
| Git | Code, schemas, experiment definitions, compact manifests and documentation | Gigabyte-scale candidate/data warehouse |

### Important anti-duplication rule

`ResearchRegistry` and `ResearchMemory` are related but not interchangeable. The Registry owns **scientific identity/governance**. Research Memory owns **search-scale empirical memory**. A future implementation should link memory records to registered experiment/run IDs rather than independently inventing competing experiment identities or authoritative verdicts.

## Implemented research pipeline

Research Pipeline V1 currently performs the following Discovery-only path:

```text
Experiment specification
        |
        v
Authorized Discovery partition
        |
        v
Partition fingerprint verification
        |
        v
Causal DiscoveryRecord population
        |
        v
Context/event-study scan
        |
        v
Descriptive metrics
        |
        v
Immutable result package
        |
        +--> experiment.json
        +--> manifest.json
        +--> metrics.json
        +--> event_study.json
        +--> report.md
        |
        v
Research Memory
```

The result status `OBSERVED` is deliberately not a profitability verdict. Screening, execution testing, robustness and protected validation are later responsibilities.

## Research Memory

Research Memory V1 is an append-oriented DuckDB store designed for the volume expected from autonomous search. It records reproducible pipeline runs and provides a candidate-memory foundation with:

- candidate fingerprint and duplicate detection;
- parent/child lineage;
- parameter payload;
- metrics payload;
- verdict state;
- explicit rejection reason for rejected candidates;
- aggregate failure summaries.

Failed candidates are useful scientific information. The system should remember why a region failed so the autonomous researcher does not blindly repeat it.

Example rejection taxonomy for later screening:

```text
REJECT_LOW_SAMPLE
REJECT_NEGATIVE_EXPECTANCY
REJECT_UNSTABLE_ACROSS_YEARS
REJECT_PARAMETER_SENSITIVE
REJECT_SESSION_DEPENDENT
REJECT_SPREAD_SENSITIVE
REJECT_OOS_FAILURE
REJECT_DUPLICATE_HYPOTHESIS
```

The exact controlled taxonomy should be frozen when Phase 2 screening is implemented.

## Example end-to-end workflow

Research question:

> Previous-day reference levels and subsequent market structure may contain predictive information.

Sandbox should not immediately hard-code a strategy such as `PDL sweep + reclaim + HL + break = BUY`. Instead the workflow is:

```text
1. HUMAN HYPOTHESIS FAMILY
   Previous-day levels + later structure may contain information

2. AVAILABLE CAUSAL FEATURES
   PDO / PDH / PDL / PDC
   distance / normalized distance
   penetration
   sweep / reclaim
   HH / HL / LH / LL
   BOS / CHoCH
   ATR / volatility / session

3. PHASE-2 CANDIDATE GENERATION
   vary level
   vary interaction type
   vary pivot left/right strength
   vary timing/session/context
   vary numeric thresholds
   combine only valid causal conditions

4. RESEARCH MEMORY CHECK
   candidate fingerprint already tested?
       YES -> skip/reuse prior evidence
       NO  -> continue

5. CHEAP DISCOVERY SCREEN
   event count
   subgroup vs complement
   forward-return behavior
   effect size
   stability diagnostics

6. FUNNEL
   50,000 generated candidates
        -> 4,000 worth deeper inspection
        -> 500 interesting candidates
        -> 80 execution backtests
        -> 15 robust candidates
        -> 5 Validation candidates
        -> 2 Validation survivors
        -> 1 Final-Test survivor

   Numbers above are illustrative, not targets or expected outcomes.

7. EXECUTION TEST
   convert surviving research condition into explicit entry/exit semantics
   apply deterministic spread, commission, slippage and gap rules

8. ROBUSTNESS
   reject fragile parameter islands
   reject insufficient samples
   reject unstable years/sessions/regimes
   reject effects destroyed by realistic costs

9. VALIDATION
   frozen candidate -> untouched Validation data

10. FINAL-TEST
   only a frozen survivor reaches the final untouched partition

11. DECISION
   REJECTED / WEAK / PROMISING / SUPPORTED / INCONCLUSIVE

12. MEMORY
   store outcome and reason

13. NEXT RESEARCH CYCLE
   analyze Discovery failures and survivors
   propose a materially justified next hypothesis
```

## Protected-data rule

The research loop is allowed to learn from **Discovery** failures and successes. If Validation results are used to redesign the hypothesis, that Validation period has become research information for the revised claim and cannot remain pristine validation for that same claim. **Final-Test evidence must never be used to tune/revise a candidate and still be called Final-Test evidence.** A revised claim requires a new untouched final source/period.

## Storage model

```text
Git
  code
  schemas
  experiment definitions
  compact manifests
  documentation
  research summaries

Parquet
  market data
  large immutable ledgers/artifacts

SQLite catalog / ResearchRegistry
  governance
  provenance
  partitions
  experiment lifecycle
  audit journal

DuckDB Research Memory
  high-volume candidate history
  metrics
  lineage
  rejection reasons
  search summaries

GitHub Actions
  regression gates
  reproducibility checks
  future manually-triggered research jobs
```

Do not commit one Git object per generated candidate. Large autonomous searches should remain queryable in DuckDB/Parquet while Git stores the definitions and compact evidence needed to reproduce them.

## Current implementation status

| Layer | Status |
|---|---|
| Market-data foundation | Implemented |
| Deterministic execution foundations | Implemented |
| Market-state / causal feature layer | Implemented |
| Configurable pivot parameters in Discovery | Implemented |
| Discovery dataset projection | Implemented |
| Research Pipeline V1 | Implemented; dedicated CI regression gate passing |
| Research Memory V1 | Implemented; dedicated CI regression gate passing |
| Autonomous candidate generation/search | **Next** |
| Progressive candidate screening/promotion | Planned |
| Search-to-backtest promotion | Planned |
| Robustness engine integration | Planned |
| Autonomous memory-driven research loop | Planned |
| Optional ML in autonomous loop | Planned |
| Final Pine/MQL5 export pipeline | Planned |

The broad legacy test suite may still contain failures outside the Phase-1/Pipeline/Memory regression gates. A green focused research gate should not be misrepresented as proof that every historical subsystem test is green.

## Phase 2 target

The next major component is the **Autonomous Candidate Search Engine**. Its contract should be:

```text
broad hypothesis/search-space definition
        |
        v
generate valid candidate specifications
        |
        v
canonical candidate fingerprint
        |
        v
Research Memory duplicate check
        |
        v
cheap event-study evaluation
        |
        v
screen / rank / reject
        |
        +--> remember every tested candidate
        |
        v
promote only survivors to expensive execution testing
```

Phase 2 should vary parameters such as pivot strength through the generic Phase-1 configuration rather than reimplementing HH/HL/LH/LL/BOS/CHoCH for each value.

## Design principles

1. **Causality first.** No feature may use information unavailable at its decision timestamp.
2. **Discovery is not validation.** Search freely only inside authorized Discovery data.
3. **Failures are evidence.** Rejected candidates belong in Research Memory.
4. **Reproducibility.** Code, dataset, partition, experiment and candidate fingerprints identify what actually ran.
5. **Progressive cost.** Cheap screens eliminate most candidates before expensive backtests.
6. **No silent mutation.** Scientific records and result artifacts are immutable or explicitly versioned.
7. **No strategy-specific core.** Reusable market primitives belong in the feature layer; strategies compose them.
8. **No automatic ML worship.** ML is retained only when protected out-of-sample evidence shows incremental value.
9. **Final means final.** Final-Test data is not a feedback source.
10. **CI is software evidence, not market evidence.** Passing tests says the implementation obeys its contract; it says nothing by itself about profitability.
