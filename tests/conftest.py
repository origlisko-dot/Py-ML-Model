"""Shared synthetic-data fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_intraday(
    freq: str = "1min",
    n_days: int = 3,
    bars_per_day: int = 120,
    start: str = "2024-06-03",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    day = pd.Timestamp(start, tz="UTC")
    price = 100.0
    made = 0
    while made < n_days:
        if day.weekday() < 5:
            idx = pd.date_range(day + pd.Timedelta(hours=14), periods=bars_per_day, freq=freq)
            steps = rng.normal(0, 0.12, bars_per_day)
            close = price + np.cumsum(steps)
            o = close - rng.normal(0, 0.03, bars_per_day)
            h = np.maximum(o, close) + np.abs(rng.normal(0, 0.05, bars_per_day))
            low = np.minimum(o, close) - np.abs(rng.normal(0, 0.05, bars_per_day))
            v = rng.integers(100, 2000, bars_per_day).astype(float)
            frames.append(
                pd.DataFrame(
                    {"open": o, "high": h, "low": low, "close": close, "volume": v},
                    index=idx,
                )
            )
            price = close[-1]
            made += 1
        day += pd.Timedelta(days=1)
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def minute_ohlcv() -> pd.DataFrame:
    """Three trading days of 1-minute OHLCV."""
    return _make_intraday(freq="1min", n_days=3, bars_per_day=120)


@pytest.fixture
def minute_ohlcv_with_hole(minute_ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Minute OHLCV with a 3-bar intraday hole punched into the first day."""
    df = minute_ohlcv.drop(minute_ohlcv.index[5:8])
    return df, 3
