from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from sandbox.execution.models import Direction, EntryType
from sandbox.research.canonical import sha256_canonical
from sandbox.strategies.models import (
    StrategyMetadata,
    StrategySignal,
    WarmupRequirements,
)

NY = ZoneInfo("America/New_York")
PIP = 0.0001

CANONICAL_MQL5_SHA256 = (
    "80736cd520f5a28836bdf271e5a01036f1e2c3fb4f208e42d612f32ada08ce05"
)


class Version0Strategy:
    metadata = StrategyMetadata(
        strategy_id="VERSION0",
        strategy_name="Version 0 NY 3H Fib",
        strategy_version="1.0.0",
        strategy_family="NY_3H_FIB",
        description=(
            "Python reproduction of canonical "
            "F1_NY3H_Green_MT5_Version0.mq5."
        ),
        supported_symbols=("EURUSD",),
        required_base_timeframe="M15",
        required_timeframes=("M15",),
        required_data_fields=("open", "high", "low", "close"),
        warmup_requirements=WarmupRequirements(
            bars_by_timeframe={"M15": 12}
        ),
        supported_directions=(Direction.LONG,),
        parameter_schema=(),
        default_parameters={},
        signal_schema="VERSION0_SIGNAL_1.0",
        plugin_version="1.0",
        capabilities=(
            "LONG",
            "LIMIT_ENTRY",
            "EXPIRING_ENTRY",
            "NY_3H_BLOCK",
            "FIXED_SL_TP",
        ),
        analysis_timezone="America/New_York",
        synthetic_test_plugin=False,
    )

    def reset(self):
        self._last_boundary = None

    def evaluate(self, context):
        m15 = context.history("EURUSD", "M15", 12)

        if len(m15) < 12:
            return ()

        now = context.current_time
        local = now.astimezone(NY)

        # NY 3-hour boundaries:
        # 00, 03, 06, 09, 12, 15, 18, 21.
        if (
            local.minute != 0
            or local.second != 0
            or local.hour % 3 != 0
        ):
            return ()

        boundary_key = local.isoformat()

        if self._last_boundary == boundary_key:
            return ()

        self._last_boundary = boundary_key

        # Previous completed 3H = 12 completed M15 candles.
        first = m15[0]
        last = m15[-1]

        block_open = float(first.values["open"])
        block_close = float(last.values["close"])

        if block_open == block_close:
            return ()

        top = max(block_open, block_close)
        bottom = min(block_open, block_close)

        midpoint = (block_open + block_close) / 2.0
        unit = top - midpoint

        if unit <= 0:
            return ()

        # Canonical Version 0 Fib -3 level.
        fib3 = midpoint - 3.0 * unit

        # Canonical risk rules.
        stop = fib3 - 40.0 * PIP
        target = fib3 + 5.0 * PIP

        # Order remains valid only for the following 3H block.
        expires_at = now + timedelta(hours=3)

        setup_id = "VERSION0-SETUP-" + sha256_canonical(
            {
                "boundary": now,
                "block_open": block_open,
                "block_close": block_close,
                "fib3": fib3,
            }
        )[:24]

        signal_id = "VERSION0-SIGNAL-" + sha256_canonical(
            {
                "setup_id": setup_id,
                "decision": now,
                "entry": fib3,
                "stop": stop,
                "target": target,
            }
        )[:24]

        return (
            StrategySignal(
                signal_id=signal_id,
                setup_id=setup_id,
                symbol="EURUSD",
                decision_timestamp=now,
                information_cutoff=context.information_cutoff,
                direction=Direction.LONG,
                entry_type=EntryType.LIMIT,
                reference_price=fib3,
                stop_price=stop,
                target_price=target,
                expires_at=expires_at,
                metadata={
                    "canonical_mql5_sha256": CANONICAL_MQL5_SHA256,
                    "ny_boundary": local.isoformat(),
                    "previous_3h_open": block_open,
                    "previous_3h_close": block_close,
                    "previous_3h_top": top,
                    "previous_3h_bottom": bottom,
                    "midpoint": midpoint,
                    "unit": unit,
                    "fib_minus_3": fib3,
                    "sl_pips": 40,
                    "tp_pips": 5,
                    "observation_hours": 3,
                },
            ),
        )