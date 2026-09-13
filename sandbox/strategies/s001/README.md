# S001 v1.0 baseline

S001 is the first real strategy plugin, implemented for deterministic specification testing only. It has not been tested for profitability and is not evidence of a trading edge.

It uses completed H1 bars for 12-bar displacement normalized by causal Wilder/RMA ATR(14), completed M15 bars for 8-bar momentum normalized by the same ATR convention, ordered impulse anchors, a 25%–60% frozen-impulse pullback, an eight-bar expiry, strict close-based resumption, structural pullback-extreme stop, MARKET_CLOSE intent, and fixed 2R target. Depth above 60% invalidates before confirmation. Parameters and status are declared in `parameters.py`.

Depth is recorded without clamping. Boundary comparisons use a fixed `1e-12` numerical tolerance so mathematically exact 25% and 60% values represented as binary floats retain their specified inclusive classification.

The state lifecycle is `WAITING_FOR_PULLBACK → PULLBACK_MONITORING/ARMED → TRADED`, with terminal `INVALID` and `EXPIRED` outcomes. One active setup per direction and one unresolved S001 position per symbol are allowed. The execution orchestrator must call `notify_position_resolved()` after Stage 2 resolves the emitted position; until then S001 emits nothing further.

Potential Discovery alternatives are documented in `FUTURE_DISCOVERY_ALTERNATIVES`. They are not executed, combined, optimized, or registered as experiments.
