"""Technical indicators and missing-bar handling.

Robustness principle #4 (missing bars): intraday series have holes when no
trade prints. Deep-learning models choke on the resulting NaNs, so
:func:`fill_missing_bars` forward-fills price *within each trading day* (never
across the overnight gap) and marks synthetic bars with ``imputed=True`` and
``volume=0`` before any indicator is computed.

Indicators are implemented directly on pandas for deterministic, dependency-light
behaviour (no reliance on TA-Lib / pandas-ta version quirks in CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.data.schemas import OHLCV_COLUMNS, normalize_ohlcv
from trading_ml.timeframes import Timeframe


# --------------------------------------------------------------------------- #
# Missing-bar handling                                                         #
# --------------------------------------------------------------------------- #
def fill_missing_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Forward-fill intraday holes at the timeframe's native frequency.

    For daily-and-coarser timeframes there is no intra-session hole concept, so
    the frame is returned unchanged (only sorted/deduped). For intraday
    timeframes, each calendar day is reindexed from its first to its last bar at
    ``timeframe`` frequency; gaps are forward-filled on ``close`` (and O/H/L set
    to that close), ``volume`` set to 0, and an ``imputed`` flag set to True.
    """
    df = normalize_ohlcv(df)
    if df.empty:
        df["imputed"] = pd.Series(dtype=bool)
        return df

    tf = Timeframe(timeframe)
    if not tf.is_intraday():
        df = df.copy()
        df["imputed"] = False
        return df

    freq = tf.pandas_freq
    filled_parts: list[pd.DataFrame] = []
    for _, day in df.groupby(df.index.normalize()):
        full = pd.date_range(day.index.min(), day.index.max(), freq=freq, tz="UTC")
        reindexed = day.reindex(full)
        imputed = reindexed["close"].isna()
        reindexed["close"] = reindexed["close"].ffill()
        for col in ("open", "high", "low"):
            reindexed[col] = reindexed[col].where(~imputed, reindexed["close"])
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
        reindexed["imputed"] = imputed.values
        filled_parts.append(reindexed)

    out = pd.concat(filled_parts).sort_index()
    out.index.name = "timestamp"
    # Any leading NaNs (day opened on an imputed bar) are dropped.
    return out.dropna(subset=["close"])


# --------------------------------------------------------------------------- #
# Indicator primitives                                                         #
# --------------------------------------------------------------------------- #
def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP (resets each trading day).

    Anchoring per day keeps VWAP meaningful for intraday timeframes and avoids
    leaking a running total across the overnight gap.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0.0, np.nan)
    return cum_pv / cum_vol


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative signed volume by close direction."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Stochastic oscillator %K and its %D smoothing (0..100)."""
    low_min = df["low"].rolling(k_window, min_periods=k_window).min()
    high_max = df["high"].rolling(k_window, min_periods=k_window).max()
    span = (high_max - low_min).replace(0.0, np.nan)
    percent_k = 100.0 * (df["close"] - low_min) / span
    percent_d = percent_k.rolling(d_window, min_periods=d_window).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI/-DI (Wilder smoothing)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr_
    minus_di = (
        100.0 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr_
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


# --------------------------------------------------------------------------- #
# Feature assembly                                                             #
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """Append a standard technical feature set to an OHLCV frame.

    Returns a new frame; the input is not mutated. NaNs from indicator warm-up
    periods are left in place — the dataset layer handles windowing.
    """
    out = df.copy()
    close = out["close"]

    out["ret_1"] = close.pct_change()
    out["log_ret_1"] = np.log(close).diff()
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["sma_20"] = sma(close, 20)
    out["rsi_14"] = rsi(close, 14)
    out = pd.concat([out, macd(close)], axis=1)
    out["atr"] = atr(out, atr_window)
    out = pd.concat([out, bollinger(close)], axis=1)
    out = pd.concat([out, stochastic(out)], axis=1)
    out = pd.concat([out, adx(out)], axis=1)
    # Volume-based features, expressed as deviations to keep them stationary.
    out["vwap_dev"] = (close - vwap(out)) / close
    obv_series = obv(out)
    out["obv_z"] = (obv_series - obv_series.rolling(50, min_periods=10).mean()) / (
        obv_series.rolling(50, min_periods=10).std().replace(0.0, np.nan)
    )
    # Normalized position of close within the bar and range vs ATR.
    span = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["close_pos"] = (close - out["low"]) / span
    out["range_atr"] = span / out["atr"].replace(0.0, np.nan)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except raw OHLCV bookkeeping."""
    exclude = set(OHLCV_COLUMNS) | {"imputed"}
    return [c for c in df.columns if c not in exclude]
