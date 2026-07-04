"""OHLCV schema and validation.

The canonical bar frame is a tz-aware (UTC) ``DatetimeIndex`` named ``timestamp``
with float columns ``open, high, low, close`` and ``volume``. An optional boolean
``imputed`` column marks bars that were forward-filled (see
``features.indicators.fill_missing_bars``).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class OHLCVSchema(pa.DataFrameModel):
    """Validated OHLCV frame. Index must be a UTC DatetimeIndex."""

    open: Series[float] = pa.Field(ge=0, nullable=False)
    high: Series[float] = pa.Field(ge=0, nullable=False)
    low: Series[float] = pa.Field(ge=0, nullable=False)
    close: Series[float] = pa.Field(ge=0, nullable=False)
    volume: Series[float] = pa.Field(ge=0, nullable=False)

    class Config:
        strict = False  # allow extra columns (e.g. `imputed`, indicators)
        coerce = True


def empty_ohlcv() -> pd.DataFrame:
    """An empty, correctly-typed OHLCV frame."""
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS}, index=idx)
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary provider frame into the canonical OHLCV shape.

    - lowercases columns, keeps only OHLCV
    - ensures a UTC tz-aware DatetimeIndex named ``timestamp``
    - sorts by time and drops duplicate timestamps (keeping the last)
    """
    if df.empty:
        return empty_ohlcv()

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    # Locate the timestamp: index or a column.
    if not isinstance(out.index, pd.DatetimeIndex):
        ts_col = next(
            (c for c in ("timestamp", "date", "datetime", "time") if c in out.columns),
            None,
        )
        if ts_col is None:
            raise ValueError("No datetime index or timestamp column found")
        out = out.set_index(ts_col)

    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = out[OHLCV_COLUMNS].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate (and coerce) a frame against :class:`OHLCVSchema`."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV frame must have a DatetimeIndex")
    return OHLCVSchema.validate(df)  # type: ignore[return-value]
