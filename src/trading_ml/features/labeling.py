"""Triple-barrier labeling (López de Prado).

For each bar we set an upper barrier (take-profit) and lower barrier (stop-loss)
placed at ATR-scaled distances, plus a vertical barrier (time horizon). The
label is which barrier is touched first, looking only *forward*:

    +1  take-profit hit first   (bullish)
    -1  stop-loss hit first     (bearish)
     0  neither within horizon  (flat / timeout)

This ties the technical model's target directly to the risk model's stop/take
geometry, and — because barriers use only future bars ``i+1 … i+horizon`` — it
is inherently point-in-time for the label itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.features.indicators import atr as atr_indicator

LABEL_COLUMNS = ["label", "ret", "touched"]


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    take_profit_atr: float = 2.0,
    stop_loss_atr: float = 1.0,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Compute triple-barrier labels on a close-labeled OHLCV frame.

    Returns a frame indexed like ``df`` with columns:
    ``label`` (∈ {-1, 0, 1}), ``ret`` (return at the touch/timeout bar), and
    ``touched`` ("tp" | "sl" | "vertical" | "" for undefined tail bars).
    Tail bars without a full horizon are labeled NaN and should be dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr = atr_indicator(df, atr_window).to_numpy(dtype=float)
    n = len(df)

    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    touched = np.array([""] * n, dtype=object)

    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or i + horizon >= n:
            continue  # warm-up or insufficient forward window
        entry = close[i]
        upper = entry + take_profit_atr * a
        lower = entry - stop_loss_atr * a

        end = i + horizon
        outcome = 0
        touch = "vertical"
        touch_price = close[end]
        for j in range(i + 1, end + 1):
            hit_tp = high[j] >= upper
            hit_sl = low[j] <= lower
            if hit_tp and hit_sl:
                # Both in same bar: assume the adverse (stop) triggers first.
                outcome, touch, touch_price = -1, "sl", lower
                break
            if hit_tp:
                outcome, touch, touch_price = 1, "tp", upper
                break
            if hit_sl:
                outcome, touch, touch_price = -1, "sl", lower
                break
        labels[i] = outcome
        rets[i] = touch_price / entry - 1.0
        touched[i] = touch

    return pd.DataFrame({"label": labels, "ret": rets, "touched": touched}, index=df.index)


def to_class_index(labels: pd.Series) -> pd.Series:
    """Map {-1, 0, +1} → {0, 1, 2} for cross-entropy classification heads."""
    return (labels.astype(int) + 1).astype("int64")
