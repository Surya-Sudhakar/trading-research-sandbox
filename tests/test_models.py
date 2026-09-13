from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from sandbox.models import Candle


def candle(**overrides):
    values = dict(timestamp_utc=datetime(2024, 1, 1, tzinfo=timezone.utc), open=Decimal("1.1"), high=Decimal("1.2"), low=Decimal("1.0"), close=Decimal("1.15"), tick_volume=2, spread=1, real_volume=0)
    values.update(overrides)
    return Candle(**values)


def test_candle_is_valid_and_immutable():
    item = candle()
    with pytest.raises(ValidationError):
        item.close = Decimal("2")


@pytest.mark.parametrize("values", [{"high": Decimal("1.0")}, {"low": Decimal("1.2")}, {"open": Decimal("0")}])
def test_invalid_candles_rejected(values):
    with pytest.raises(ValidationError):
        candle(**values)

