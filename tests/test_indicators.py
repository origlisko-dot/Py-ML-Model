"""Robustness principle #4: missing-bar forward-fill."""

import numpy as np

from trading_ml.features.indicators import (
    add_indicators,
    atr,
    feature_columns,
    fill_missing_bars,
    rsi,
)


def test_forward_fill_recovers_hole(minute_ohlcv_with_hole):
    df, hole = minute_ohlcv_with_hole
    filled = fill_missing_bars(df, "1m")

    # The 3-bar hole is recovered and flagged.
    assert int(filled["imputed"].sum()) >= hole
    # No NaNs remain in price.
    assert not filled[["open", "high", "low", "close"]].isna().any().any()
    # Imputed bars carry zero volume and O/H/L equal to the carried close.
    imp = filled[filled["imputed"]]
    assert (imp["volume"] == 0).all()
    assert np.allclose(imp["open"], imp["close"])
    assert np.allclose(imp["high"], imp["close"])


def test_no_fill_across_overnight_gap(minute_ohlcv):
    # Filling must not invent bars between trading days.
    filled = fill_missing_bars(minute_ohlcv, "1m")
    days = filled.index.normalize().nunique()
    assert days == 3
    # total bars should not explode to a continuous multi-day minute grid
    assert len(filled) < 3 * 24 * 60


def test_daily_timeframe_not_filled(minute_ohlcv):
    daily = (
        minute_ohlcv.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    out = fill_missing_bars(daily, "1d")
    assert "imputed" in out.columns
    assert not out["imputed"].any()


def test_indicators_present_and_finite(minute_ohlcv):
    filled = fill_missing_bars(minute_ohlcv, "1m")
    feats = add_indicators(filled)
    cols = feature_columns(feats)
    assert {"rsi_14", "atr", "macd", "bb_width"}.issubset(set(cols))
    # After warm-up, indicators are finite.
    tail = feats[cols].iloc[50:]
    assert np.isfinite(tail.to_numpy()).mean() > 0.9


def test_rsi_bounds(minute_ohlcv):
    r = rsi(minute_ohlcv["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_atr_positive(minute_ohlcv):
    a = atr(minute_ohlcv, 14).dropna()
    assert (a >= 0).all()
