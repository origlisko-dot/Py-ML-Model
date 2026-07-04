"""Multi-timeframe resampling and strict point-in-time alignment.

Robustness principle #2 (no lookahead) — the single most important correctness
rule in the codebase:

Provider bars are **start-labeled** (a 5-minute bar stamped 10:00 covers
``[10:00, 10:05)`` and is only *known* at 10:05). Every frame here is converted
to **close-time labels** so that a bar's index is the instant it becomes
available. Higher-timeframe context is then joined with a *backward* as-of merge:
a decision at time ``T`` sees a higher-timeframe bar only once that bar has
closed (its close-label ``<= T``). A 1-hour bar covering ``[10:00, 11:00)`` is
therefore invisible to a 10:15 decision — it only appears from 11:00 onward.
"""

from __future__ import annotations

import pandas as pd

from trading_ml.features.candlestick import add_patterns
from trading_ml.features.indicators import add_indicators, fill_missing_bars
from trading_ml.timeframes import Timeframe

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _offset(timeframe: str):
    return pd.tseries.frequencies.to_offset(Timeframe(timeframe).pandas_freq)


def to_close_labeled(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Shift a start-labeled frame to close-time labels (index += one bar).

    After this, ``index`` is the moment each bar becomes known — the anchor for
    all point-in-time comparisons.
    """
    out = df.copy()
    out.index = out.index + _offset(timeframe)
    out.index.name = "timestamp"
    return out


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Aggregate a finer, start-labeled OHLCV frame to a coarser timeframe.

    The result is **close-labeled**: each aggregated bar is stamped at the end
    of its interval, i.e. when it becomes known.
    """
    freq = Timeframe(target_timeframe).pandas_freq
    agg = df.resample(freq, label="left", closed="left").agg(_AGG).dropna(subset=["close"])
    agg.index = agg.index + _offset(target_timeframe)
    agg.index.name = "timestamp"
    return agg


def build_timeframe_features(
    base_df: pd.DataFrame,
    timeframe: str,
    base_timeframe: str,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Build a close-labeled feature frame for one context timeframe.

    ``base_df`` is the (start-labeled) base-timeframe OHLCV. If ``timeframe``
    equals ``base_timeframe`` the frame is only relabeled to close time;
    otherwise it is resampled up first. Missing bars are forward-filled before
    indicators/patterns are computed.
    """
    if timeframe == base_timeframe:
        closed = to_close_labeled(fill_missing_bars(base_df, base_timeframe), base_timeframe)
    else:
        # fill holes at the base frequency, then aggregate to the target tf.
        filled = fill_missing_bars(base_df, base_timeframe)
        closed = resample_ohlcv(filled[["open", "high", "low", "close", "volume"]], timeframe)

    feats = add_indicators(closed, atr_window=atr_window)
    feats = add_patterns(feats)
    return feats


def align_point_in_time(
    base_index: pd.DatetimeIndex,
    higher: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    """Backward as-of join of a higher-timeframe frame onto ``base_index``.

    For each timestamp in ``base_index`` the most recent *already-closed*
    higher-timeframe row is attached. No future information can leak because
    both frames are close-labeled and the merge direction is ``backward``.
    """
    left = pd.DataFrame(index=pd.DatetimeIndex(base_index, name="timestamp")).reset_index()
    right = higher.add_suffix(f"_{suffix}").reset_index()
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.set_index("timestamp")


def build_mtf_matrix(
    base_df: pd.DataFrame,
    context_timeframes: list[str],
    base_timeframe: str,
    atr_window: int = 14,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (base_close_labeled_features, {tf: close_labeled_features}).

    The base features drive the sample index; the per-timeframe frames are used
    by the dataset to slice as-of windows (see ``models.technical.dataset``).
    Every frame is point-in-time correct.
    """
    per_tf: dict[str, pd.DataFrame] = {}
    for tf in context_timeframes:
        per_tf[tf] = build_timeframe_features(base_df, tf, base_timeframe, atr_window)
    base_feats = per_tf.get(base_timeframe)
    if base_feats is None:
        base_feats = build_timeframe_features(base_df, base_timeframe, base_timeframe, atr_window)
        per_tf[base_timeframe] = base_feats
    return base_feats, per_tf
