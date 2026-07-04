"""yfinance market-data provider — the default for development and CI.

Requires no broker or gateway. Intraday history is limited by Yahoo (roughly
1m ≤ 30d, other intraday ≤ 60d); callers should clamp windows accordingly.
"""

from __future__ import annotations

import pandas as pd

from trading_ml.data.providers.base import BaseMarketDataProvider
from trading_ml.data.schemas import empty_ohlcv, normalize_ohlcv
from trading_ml.timeframes import Timeframe

# Map canonical timeframe -> yfinance interval string.
_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
}


class YFinanceProvider(BaseMarketDataProvider):
    name = "yfinance"

    def __init__(self, **_: object) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "yfinance is required. Install with `uv sync --extra providers`."
            ) from exc

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        import yfinance as yf

        interval = _INTERVAL.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")

        raw = yf.download(
            tickers=symbol,
            start=pd.Timestamp(start).tz_convert("UTC").tz_localize(None),
            end=pd.Timestamp(end).tz_convert("UTC").tz_localize(None),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return empty_ohlcv()

        # yfinance may return a column MultiIndex (field, ticker) for one symbol.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns=str.lower)
        # Prefer raw close over adjusted for intraday pattern fidelity.
        if "adj close" in raw.columns:
            raw = raw.drop(columns=["adj close"])

        _ = Timeframe(timeframe)  # validate label
        return normalize_ohlcv(raw)
