"""Candlestick pattern detection.

Vectorized, dependency-light detectors returning integer flags (0/1, or ±1 for
directional patterns) aligned to the bar on which the pattern *completes*. These
become model features alongside the numeric indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PATTERN_COLUMNS = [
    "cdl_doji",
    "cdl_hammer",
    "cdl_inverted_hammer",
    "cdl_engulfing",  # +1 bullish, -1 bearish
    "cdl_shooting_star",
    "cdl_marubozu",  # +1 bullish, -1 bearish
    "cdl_morning_evening_star",  # +1 morning (bullish), -1 evening (bearish)
]


def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    return (c - o).abs()


def _range(h: pd.Series, low: pd.Series) -> pd.Series:
    return (h - low).replace(0.0, np.nan)


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of candlestick pattern flags indexed like ``df``."""
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    body = _body(o, c)
    rng = _range(h, low)
    upper_shadow = h - o.combine(c, max)
    lower_shadow = o.combine(c, min) - low
    bullish = c > o
    bearish = c < o

    flags = pd.DataFrame(index=df.index)

    # Doji: tiny body relative to range.
    flags["cdl_doji"] = (body <= 0.1 * rng).astype("int8")

    # Hammer: small body near top, long lower shadow, small upper shadow.
    flags["cdl_hammer"] = (
        (lower_shadow >= 2.0 * body) & (upper_shadow <= body) & (body > 0)
    ).astype("int8")

    # Inverted hammer: long upper shadow, small lower shadow.
    flags["cdl_inverted_hammer"] = (
        (upper_shadow >= 2.0 * body) & (lower_shadow <= body) & (body > 0)
    ).astype("int8")

    # Shooting star: like inverted hammer but bearish context (body at low end).
    flags["cdl_shooting_star"] = (
        (upper_shadow >= 2.0 * body) & (lower_shadow <= 0.3 * body) & bearish
    ).astype("int8")

    # Marubozu: body ~ full range (negligible shadows).
    marubozu = body >= 0.9 * rng
    flags["cdl_marubozu"] = np.where(
        marubozu & bullish, 1, np.where(marubozu & bearish, -1, 0)
    ).astype("int8")

    # Engulfing (2-bar): current body engulfs previous body, opposite colour.
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bull = prev_c > prev_o
    prev_bear = prev_c < prev_o
    bull_engulf = bullish & prev_bear & (c >= prev_o) & (o <= prev_c)
    bear_engulf = bearish & prev_bull & (o >= prev_c) & (c <= prev_o)
    flags["cdl_engulfing"] = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0)).astype("int8")

    # Morning/Evening star (3-bar): big body, small body (gap), big opposite body.
    b1 = body.shift(2)
    b2 = body.shift(1)
    o0, c0 = o, c
    c2 = c.shift(2)
    mid2 = (o.shift(2) + c.shift(2)) / 2.0
    small_middle = b2 <= 0.5 * b1
    morning = (
        (c.shift(2) < o.shift(2))  # bar-2 bearish
        & small_middle
        & bullish  # bar-0 bullish
        & (c0 > mid2)
    )
    evening = (
        (c.shift(2) > o.shift(2))  # bar-2 bullish
        & small_middle
        & bearish  # bar-0 bearish
        & (c0 < mid2)
    )
    flags["cdl_morning_evening_star"] = np.where(morning, 1, np.where(evening, -1, 0)).astype(
        "int8"
    )

    _ = (b1, b2, o0, c2)  # keep intermediate names explicit for readability
    return flags[PATTERN_COLUMNS].fillna(0).astype("int8")


def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Append candlestick pattern flags to ``df`` (returns a new frame)."""
    return pd.concat([df, detect_patterns(df)], axis=1)
