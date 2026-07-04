import numpy as np
import pandas as pd

from trading_ml.features.labeling import to_class_index, triple_barrier_labels


def _frame_from_close(closes, highs=None, lows=None):
    idx = pd.date_range("2024-06-03 14:00", periods=len(closes), freq="5min", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    highs = closes if highs is None else np.asarray(highs, dtype=float)
    lows = closes if lows is None else np.asarray(lows, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1.0},
        index=idx,
    )


def test_take_profit_label():
    # Rising series should hit the upper barrier -> label +1.
    closes = np.linspace(100, 110, 60)
    df = _frame_from_close(closes, highs=closes + 0.5, lows=closes - 0.1)
    lab = triple_barrier_labels(
        df, horizon=10, take_profit_atr=1.0, stop_loss_atr=1.0, atr_window=5
    )
    valid = lab.dropna()
    assert (valid["label"] == 1).mean() > 0.6


def test_stop_loss_label():
    closes = np.linspace(110, 100, 60)
    df = _frame_from_close(closes, highs=closes + 0.1, lows=closes - 0.5)
    lab = triple_barrier_labels(
        df, horizon=10, take_profit_atr=1.0, stop_loss_atr=1.0, atr_window=5
    )
    valid = lab.dropna()
    assert (valid["label"] == -1).mean() > 0.6


def test_labels_in_domain_and_tail_undefined(minute_ohlcv):
    lab = triple_barrier_labels(minute_ohlcv, horizon=8, atr_window=14)
    valid = lab.dropna()
    assert set(valid["label"].unique()).issubset({-1.0, 0.0, 1.0})
    # last `horizon` bars cannot be labeled
    assert lab["label"].iloc[-1] != lab["label"].iloc[-1] or np.isnan(lab["label"].iloc[-1])
    assert set(to_class_index(valid["label"]).unique()).issubset({0, 1, 2})
