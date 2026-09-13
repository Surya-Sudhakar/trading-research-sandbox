# Universal Feature Engine V1

The existing Trading Research Sandbox now exposes a deterministic batch API:

```python
from sandbox.market_state.universal import UniversalFeatureEngine

engine = UniversalFeatureEngine()  # M15, existing America/New_York analysis timezone
rows = engine.compute(completed_bars, symbol="EURUSD")
# Or provide an explicit timezone-aware as_of to filter unfinished input bars.
rows = engine.compute(all_bars, symbol="EURUSD", as_of=cutoff)
records = [row.model_dump(mode="json") for row in rows]
metadata = engine.definitions
```

Input is an ordered pandas DataFrame with timestamp_utc/open/high/low/close,
or an iterable of the existing sandbox.models.Candle. Input timestamps are UTC
bar opens (other aware zones are converted); output timestamp is UTC bar close.
The caller must supply completed bars or an explicit as_of cutoff, and must
obtain data through the existing authorized research-partition path. This pure
function does not read data, authorize access, or update ledgers.

## Architecture and reuse

This is an additive subpackage of sandbox.market_state. Existing strategy,
execution, CLI and MarketStateEngine behavior is unchanged. The project CLI is
argparse-based; no new command or UI is needed for this programmatic task.

Candle geometry, volatility, momentum, structure, context and schema each have
their own module. FeatureRow is frozen and typed. definitions() exposes an
immutable tuple with name/category/dtype/timeframe/lookback/description/version.
Lookback means minimum observed bars including the current bar, measured in
the definition's timeframe; ATR remains recursive over the supplied history.

Reuses market_state true_ranges, causal_percentile, safe_ratio,
timeframe_minutes and FeatureConfiguration.analysis_timezone. The existing
PartitionService._strategy_timeframe_frames reconstructs H1/H3 with UTC floors
and requires every source slot. Input grid, ordering and uniqueness validation
make its row-count completeness rule safe here. No synthetic prices are filled.
Its metadata adapter requests only timeframes; no strategy is instantiated.

The old trailing_atr_series repeatedly calculates all prefixes. This package
uses the identical seed/recurrence in one pass, avoiding quadratic work.
An automated test compares it to the existing wilder_atr for every prefix.

## Definitions and warm-up

Let R=high-low, B=abs(close-open), and t be the current completed bar.

| Feature | Definition | First possible bar |
|---|---|---:|
| candle_range | R | 1 |
| body_size | B | 1 |
| body_ratio | B/R; null for R=0 | 1 |
| upper_wick | high-max(open,close) | 1 |
| lower_wick | min(open,close)-low | 1 |
| body_midpoint | (open+close)/2 | 1 |
| close_location | (close-low)/R; null for R=0 | 1 |
| candle_direction | sign(close-open) | 1 |
| atr_14 | Wilder recurrence described below | 15 |
| atr_percentile_100 | Inclusive weak rank of ATR over 100 trailing valid ATR values | 114 |
| range_percentile_100 | Inclusive weak rank of R over 100 trailing bars | 100 |
| displacement_4_atr | (close[t]-close[t-4])/ATR[t] | 15 |
| displacement_8_atr | (close[t]-close[t-8])/ATR[t] | 15 |
| prior_high_12 | max(high[t-12],...,high[t-1]) | 13 |
| prior_low_12 | min(low[t-12],...,low[t-1]) | 13 |
| breakout_high_12 | close[t]>prior_high_12; equality false | 13 |
| breakout_low_12 | close[t]<prior_low_12; equality false | 13 |
| completed_h1_direction | sign(close-open) of latest H1 ending at or before decision | 1 complete H1 |
| completed_h3_direction | sign(close-open) of latest H3 ending at or before decision | 1 complete H3 |
| analysis_hour | decision close hour in configured analysis timezone | 1 |

TR[t] = max(high[t]-low[t], abs(high[t]-close[t-1]),
abs(low[t]-close[t-1])). The first input bar has no TR. ATR on bar 15 is the
mean of TRs on bars 2..15. Thereafter ATR=(13*previous_ATR+current_TR)/14.
Nonpositive ATR is null, matching existing project semantics. Its internal
zero recurrence is retained so later nonzero ranges recover correctly.

Weak rank = 100 * count(window values <= current) / 100. The current
observation is included; ties count fully. A complete valid 100-observation
window is required. No shorter-window rank is fabricated.

feature_ready is true only when all 20 values are available. The first
possible ready row is bar 114, subject to complete context and valid nonzero
denominators. False breakout flags and doji direction 0 remain valid values.
Insufficient histories, zero denominators and missing context use JSON null.

## Leakage and input rules

- Strictly increasing, unique, timezone-aware, grid-aligned bar opens.
- Finite positive OHLC with consistent high/low bounds.
- Current completed decision bar is allowed; future decision bars are not used.
- Prior structure excludes the current candle; all ranks are trailing.
- Context is exposed only when aggregate close <= decision close.
- Partial start/end buckets and buckets with missing source slots are dropped.
- Latest older completed context remains available through gaps.
- No labels, outcomes or caller-supplied feature columns enter calculations.
- Negative shifts, centered windows and future-value filling are absent.

Histories count observed bars across market gaps; they are not elapsed-clock
windows. Context can be stale across gaps. ATR depends on the supplied
historical start, as does the existing implementation; retain identical
history/configuration for exact reproducibility.

V1 accepts any symbol and fixed decision timeframes dividing both H1 and H3
(e.g. M1/M5/M15/M30/H1). Use project spelling H3, not 3H. Other context
timeframes require a future schema extension. H3 remains UTC anchored even
when the display/analysis timezone changes.

Memory is O(N); fixed-size rank/structure windows and linear ATR avoid O(N^2).
The existing reconstruction groups source bars once per context timeframe.
Each compute call owns its state, permitting independent future parallel jobs.

## Verification

Run from the existing repository:

```powershell
python -m pytest tests/test_universal_features.py -p no:cacheprovider
python -m pytest --ignore=tests/runtime -p no:cacheprovider
```

tests/runtime contains generated artifacts, not test source. Old inaccessible
directories there can prevent default pytest collection. No source tests are
excluded by the command above. The new suite covers geometry, independently
calculated ATR and displacement, weak ranks, strict breakouts, UTC H1/H3
boundaries, gaps, future mutations, prefix equivalence, deterministic JSON,
warm-up, DST, typed model input, alternate symbols/timeframes and invalid data.

Historical sample limitation: the catalog's EURUSD M15 external import is
QUARANTINED; the accepted EURUSD M1 dataset contains only 61 bars. Neither is
suitable for five fully warmed M15 rows. The delivered sample is explicitly
synthetic and is not market research evidence.

Recommended next step: repair/validate historical-data ingestion and produce
a provenance-bound feature dataset through the existing Discovery access
path. Strategy generation and Step 2 are not implemented.

