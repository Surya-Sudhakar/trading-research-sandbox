# Autonomous Trading R&D Sandbox

A reproducible research system for discovering, testing, rejecting, validating, and eventually exporting systematic trading strategies.

The core principle is simple: **a trading idea is not evidence of an edge**. Sandbox separates market data, causal features, discovery, execution simulation, research memory, validation, and final holdout testing.

> **Current state:** the market-data foundation, causal feature layer, Discovery Dataset, Research Pipeline V1, and Research Memory V1 are implemented. Autonomous candidate generation/search is the next major layer. Passing software tests does not imply a profitable trading strategy.

## Architecture

```text
Historical market data
        |
Data integrity + provenance
        |
Causal market-state / feature engine
        |
Discovery Dataset
        |
Research Pipeline V1                 [implemented]
        |
Research Memory V1                   [implemented]
        |
Autonomous candidate search          [next]
        |
Event-study screening
        |
Deterministic execution/backtesting
        |
Robustness screening
        |
Validation Guardian
Discovery -> Validation -> Final Holdout
        |
Research verdict + memory
        |
Autonomous research loop
        |
Strategy DSL / Pine / MQL5 export
```

See **[docs/OVERALL_ARCHITECTURE.md](docs/OVERALL_ARCHITECTURE.md)** for the complete architecture, ownership boundaries, storage model, research funnel, and worked example. Scientific isolation rules are in **[RESEARCH_RULES.md](RESEARCH_RULES.md)**.

## Current status

| Component | Status | Responsibility |
| --- | --- | --- |
| Market-data foundation | Implemented | Historical ingestion, provenance, integrity, immutable artifacts |
| External data gateway | Implemented | Controlled historical file ingestion and certification |
| Deterministic execution engine | Implemented | Historical fills, exits, costs and trade evidence |
| Strategy plugin interface | Implemented | Controlled deterministic strategy components |
| Market-state / Feature Engine | Implemented | Causal ATR, volatility, pivots, structure, levels and predictors |
| Discovery Dataset | Implemented | Predictor/target research surface with no-lookahead controls |
| Research Pipeline V1 | Implemented; focused CI passing | Reproducible Discovery experiments and result packages |
| Research Memory V1 | Implemented; focused CI passing | Candidate/run memory, lineage, duplicate detection and failure reasons |
| Autonomous candidate search | Next | Generate and screen parameter/condition combinations |
| Robustness promotion funnel | Planned | Stability, sensitivity and friction screening |
| Autonomous research loop | Planned | Use accumulated evidence to choose subsequent searches |
| Strategy export | Planned | Compile supported strategies toward DSL/Pine/MQL5 |

The broad legacy test suite still contains failures outside the focused Phase-1/Pipeline/Memory gates. Repository-wide CI status must not be interpreted as evidence for or against a trading strategy.

## Research model

Sandbox has three distinct responsibilities. **Market understanding** converts historical observations into causal reusable features. **Research infrastructure** executes hypotheses, measures outcomes, simulates execution when appropriate, and preserves reproducibility evidence. **Autonomous research** will generate candidate conditions and parameters, consult Research Memory, reject duplicates, screen cheaply before expensive tests, and use accumulated failures and survivors to decide where to search next.

```text
Broad hypothesis
      |
Generate candidate variants
      |
Check Research Memory -> skip duplicates
      |
Cheap event-study screening
      |
Backtest survivors
      |
Robustness checks
      |
Validation
      |
Final untouched holdout
      |
SUPPORTED / REJECTED / WEAK / INCONCLUSIVE
      |
Store evidence + failure reason
      |
Next research question
```

## System-of-record boundaries

| Store | Owns | Does not own |
| --- | --- | --- |
| Data catalog + Parquet | Dataset provenance and market-data artifacts | Research memory |
| Research Registry / SQLite | Hypotheses, experiments, lifecycle, authorization, audit history | Candidate-search cache |
| Research Memory / DuckDB | Tested candidates, parameters, lineage, metrics, verdicts, rejection reasons | Experiment governance |
| Execution ledgers / Parquet | Trade-by-trade simulation evidence | Hypothesis registry |
| Git | Source, experiment definitions, docs, manifests, compact research history | Bulk candidate database |

Research Registry and Research Memory are therefore complementary rather than duplicate databases: the Registry answers **what experiment was authorized and what happened to it?** Research Memory answers **what has already been tried and what was learned?**

## Example workflow

A human might provide only this broad hypothesis:

> Previous-day reference levels and subsequent market structure may contain predictive information.

Sandbox should not immediately hard-code a rule such as `PDL sweep + reclaim + HL = BUY`. The system exposes causal previous-day and structure features, varies eligible parameters, generates candidate conditions, screens them on Discovery data, remembers every tested candidate, and promotes only evidence-supported regions.

A future autonomous search can investigate combinations involving previous-day high/low/close, distance and penetration, sweep/reclaim state, pivot configuration, HH/HL/LH/LL, BOS/CHoCH, volatility and session context. The search space is reduced progressively through validity constraints, event studies, backtests, robustness checks, Validation, and finally an untouched holdout.

## Data and causality

Market data can enter through the read-only IC Markets MT5 history path or the controlled external historical-data gateway. External imports require explicit provenance and interpretation metadata; preserved artifacts are fingerprinted.

The research layer is causal by design. Completed higher-timeframe bars are used only when available, pivots remain invisible until confirmation, structure events use confirmed evidence, and protected Validation/Final data remain behind controlled boundaries. Missing candles are not silently synthesized.

Large market datasets and large candidate histories should not be ordinary Git history. Git is the source of truth for code, experiment specifications, manifests and compact summaries; Parquet and DuckDB are the bulk stores.

## Execution and strategies

The deterministic execution engine receives trade intents; it does not discover signals. It models entry timing, stops, targets, gaps, spread, commission and slippage with explicit rules and produces immutable trade-ledger evidence. Discovery and execution remain separate: an interesting market effect must be demonstrated before it is promoted into an execution rule.

Strategies live behind `sandbox.strategies` and declare metadata, causal timeframes, required fields, warm-up, directions, parameters and signal schema. Strategy-specific market logic does not belong in the sandbox core.

`S001` is an implementation example and has been software-tested with synthetic fixtures. That is **not evidence of profitability or a validated edge**. Its detailed assumptions belong in [sandbox/strategies/s001/README.md](sandbox/strategies/s001/README.md), not in this root README.

## Local UI and API

The React/TypeScript frontend in `frontend/` is a view/control surface over the Python implementation, not a second research engine. The controlled FastAPI adapter prevents the browser from directly accessing databases, Parquet, MT5, protected partitions, research journals or the Final-Test vault.

```powershell
# Terminal 1 - project root
python -m sandbox api serve

# Terminal 2
cd frontend
npm install
npm run dev
```

The intended deployment remains local.

## Installation

Windows is required for the IC Markets MT5 integration. Python 3.12+ is supported.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `SANDBOX_MT5_TERMINAL_PATH` only if automatic terminal discovery does not select the intended terminal.

## Common commands

```powershell
python -m sandbox mt5 status
python -m sandbox symbols search EURUSD
python -m sandbox history inspect EURUSD
python -m sandbox data catalog

python -m sandbox strategies list
python -m sandbox strategies inspect SYNTHETIC_INTERFACE_TEST
python -m sandbox strategies validate SYNTHETIC_INTERFACE_TEST

python -m sandbox.research.research_pipeline experiments/example_previous_day_context.json

pytest
```

A research-pipeline experiment must reference a real authorized Discovery partition. The example specification intentionally contains a placeholder partition ID.

## Repository map

```text
sandbox/
  market_state/        causal feature and Discovery context machinery
  research/            pipeline, registry, event studies and Research Memory
  partition/           Discovery/Validation/Final partition controls
  strategies/          strategy plugins and DSL-related components
  execution/           deterministic historical execution
  api/                 controlled local API

experiments/            versioned research experiment specifications
docs/                   architecture and technical documentation
frontend/               local interface
data/                    governed market-data artifacts and metadata
results/                 generated research/execution outputs
.github/workflows/       focused and diagnostic CI workflows
```

## Research integrity

```text
DISCOVERY
   |
Generate and modify hypotheses here
   |
VALIDATION
   |
Independent challenge of frozen candidates
   |
FINAL HOLDOUT
   |
Untouched confirmation layer
```

If Validation information is used to redesign a candidate, it has become research information for the revised claim. If Final-Holdout information is used to optimize a strategy, it is no longer a final holdout for that claim.

A research failure is not a pipeline failure. A technically successful run can correctly reject a candidate for negative expectancy, inadequate sample size, instability, session dependence, excessive friction sensitivity or out-of-sample failure. Those failures are useful Research Memory.

## Safety boundary

The historical MT5 adapter is intentionally read-only and exposes no live order-placement API. Research/simulation remains separate from live broker execution unless a future execution boundary is explicitly designed and reviewed.

This repository is research software, not a claim of profitability and not financial advice.

## Documentation

- **[Overall Architecture](docs/OVERALL_ARCHITECTURE.md)** - complete system architecture, responsibilities, workflow and roadmap.
- **[Research Rules](RESEARCH_RULES.md)** - scientific governance and protected-data rules.
- **[S001 documentation](sandbox/strategies/s001/README.md)** - S001-specific implementation assumptions.
- **[CI notes](.github/workflows/README.md)** - workflow/test-gate notes.

Historical implementation details belong with their subsystem documentation or in Git history rather than accumulating indefinitely in the root README.
