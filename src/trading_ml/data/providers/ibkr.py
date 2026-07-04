"""Interactive Brokers market-data provider (via ``ib_async``).

Requires a running TWS or IB Gateway. **Use a paper account for research.**

Pacing: IBKR rejects rapid or oversized historical requests
("pacing violations"). This provider enforces a *minimum spacing* between
requests internally; :mod:`trading_ml.data.ingestion` additionally chunks the
window and checkpoints each chunk, so a dropped connection resumes rather than
restarting.
"""

from __future__ import annotations

import time

import pandas as pd

from trading_ml.config import load_config
from trading_ml.data.providers.base import BaseMarketDataProvider
from trading_ml.data.schemas import empty_ohlcv, normalize_ohlcv

# Canonical timeframe -> IBKR barSizeSetting.
_BAR_SIZE = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week",
    "1M": "1 month",
}


def _duration_str(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """IBKR durationString covering ``[start, end]`` (rounded up to days)."""
    days = max(1, int((end - start) / pd.Timedelta(days=1)) + 1)
    if days <= 365:
        return f"{days} D"
    years = (days // 365) + 1
    return f"{years} Y"


class IBKRProvider(BaseMarketDataProvider):
    name = "ibkr"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        min_spacing_seconds: float = 10.0,
        **_: object,
    ) -> None:
        cfg = load_config()
        self.host = host or cfg.ibkr.host
        self.port = port or cfg.ibkr.port
        self.client_id = client_id or cfg.ibkr.client_id
        self.min_spacing_seconds = min_spacing_seconds
        self._ib = None
        self._last_request_ts = 0.0

    # -- connection ------------------------------------------------------- #
    def _connect(self):
        if self._ib is not None and self._ib.isConnected():
            return self._ib
        try:
            from ib_async import IB
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ib_async is required for IBKR. Install with "
                "`uv sync --extra providers` and run TWS/IB Gateway."
            ) from exc
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
        self._ib = ib
        return ib

    def _throttle(self) -> None:
        """Block until the minimum spacing since the last request has elapsed."""
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.min_spacing_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- fetch ------------------------------------------------------------ #
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        from ib_async import Stock, util

        bar_size = _BAR_SIZE.get(timeframe)
        if bar_size is None:
            raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

        ib = self._connect()
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        self._throttle()
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=pd.Timestamp(end).tz_convert("UTC").to_pydatetime(),
            durationStr=_duration_str(start, end),
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,  # UTC epoch
        )
        if not bars:
            return empty_ohlcv()

        df = util.df(bars)
        if df is None or df.empty:
            return empty_ohlcv()
        df = df.rename(columns={"date": "timestamp"})
        return normalize_ohlcv(df)

    def close(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None
