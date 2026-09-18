import math
import pytest
from sandbox.market_state.universal.market_metrics import (
    atr_change, efficiency_ratio, normalized_regression_slope,
    prior_range_position, realized_volatility,
)


def test_efficiency_ratio_extremes_and_warmup():
    trend = [1, 2, 3, 4, 5]
    chop = [1, 2, 1, 2, 1]
    assert efficiency_ratio(trend, 3, 4) is None
    assert efficiency_ratio(trend, 4, 4) == 1.0
    assert efficiency_ratio(chop, 4, 4) == 0.0
    assert efficiency_ratio([1]*5, 4, 4) == 0.0


def test_realized_volatility_matches_definition():
    c = [100, 101, 100, 102]
    expected = math.sqrt(sum(math.log(c[i]/c[i-1])**2 for i in range(1, 4)))
    assert realized_volatility(c, 3, 3) == pytest.approx(expected)
    assert realized_volatility(c, 2, 3) is None


def test_atr_change_is_causal_ratio():
    atr = [None, 2.0, 2.5, 3.0]
    assert atr_change(atr, 3, 2) == pytest.approx(0.5)
    assert atr_change(atr, 2, 2) is None


def test_normalized_regression_slope_direction_and_scale():
    assert normalized_regression_slope([1,2,3,4], 3, 4, 2.0) == pytest.approx(0.5)
    assert normalized_regression_slope([4,3,2,1], 3, 4, 2.0) == pytest.approx(-0.5)
    assert normalized_regression_slope([1,2,3], 2, 4, 2.0) is None


def test_prior_range_position_excludes_current_bar_and_allows_breakouts():
    highs = [10, 10, 10, 999]
    lows = [5, 5, 5, 1]
    assert prior_range_position(highs, lows, 7.5, 3, 3) == pytest.approx(0.5)
    assert prior_range_position(highs, lows, 11, 3, 3) > 1
    assert prior_range_position([5,5,5,9], [5,5,5,1], 5, 3, 3) == 0.5
