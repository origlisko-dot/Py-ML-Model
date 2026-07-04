import pandas as pd
import pytest

from trading_ml.timeframes import (
    Timeframe,
    duration_to_timedelta,
    sort_labels,
    to_freq,
)


def test_parse_and_freq():
    assert to_freq("5m") == "5min"
    assert to_freq("1h") == "1h"
    assert to_freq("1d") == "1D"
    assert Timeframe("15m").seconds == 900


def test_invalid_timeframe():
    with pytest.raises(ValueError):
        Timeframe("7q")


def test_ordering():
    assert sort_labels(["1d", "1m", "1h", "5m"]) == ["1m", "5m", "1h", "1d"]
    assert Timeframe("1h").is_coarser_than(Timeframe("5m"))
    assert Timeframe("1m").is_intraday()
    assert not Timeframe("1d").is_intraday()


def test_duration_to_timedelta():
    assert duration_to_timedelta("30d") == pd.Timedelta(days=30)
    assert duration_to_timedelta("2w") == pd.Timedelta(days=14)
