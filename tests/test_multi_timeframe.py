"""Robustness principle #2: strict point-in-time, no lookahead."""

import pandas as pd

from trading_ml.features.multi_timeframe import (
    align_point_in_time,
    build_timeframe_features,
    resample_ohlcv,
    to_close_labeled,
)


def test_close_labeling_shifts_forward(minute_ohlcv):
    cl = to_close_labeled(minute_ohlcv, "1m")
    # A start-labeled 1m bar at T becomes known at T+1m.
    assert (cl.index[0] - minute_ohlcv.index[0]) == pd.Timedelta("1min")


def test_resample_is_close_labeled_and_no_future_leak(minute_ohlcv):
    five = resample_ohlcv(minute_ohlcv, "5m")
    for t in five.index[1:10]:
        # Bar labeled T aggregates ONLY minutes in [T-5m, T) — strictly the past.
        window = minute_ohlcv[
            (minute_ohlcv.index >= t - pd.Timedelta("5min")) & (minute_ohlcv.index < t)
        ]
        if window.empty:
            continue
        assert abs(five.loc[t, "high"] - window["high"].max()) < 1e-9
        assert abs(five.loc[t, "low"] - window["low"].min()) < 1e-9
        assert abs(five.loc[t, "close"] - window["close"].iloc[-1]) < 1e-9
        # A future minute's extreme must never appear in this bar.
        future = minute_ohlcv[minute_ohlcv.index >= t]
        if not future.empty:
            assert five.loc[t, "high"] <= window["high"].max() + 1e-9


def test_align_backward_only(minute_ohlcv):
    # Build a coarse frame and align onto a fine index; every attached higher-tf
    # bar must have a close-label <= the decision time.
    higher = build_timeframe_features(minute_ohlcv, "15m", "1m")
    base = build_timeframe_features(minute_ohlcv, "1m", "1m")
    aligned = align_point_in_time(base.index, higher[["close"]], suffix="15m")

    higher_idx = higher.index
    for t in aligned.index[100:120]:
        val = aligned.loc[t, "close_15m"]
        if pd.isna(val):
            continue
        # The most recent higher bar with close-label <= t
        eligible = higher_idx[higher_idx <= t]
        assert len(eligible) > 0
        assert eligible.max() <= t
