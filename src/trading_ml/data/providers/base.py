"""Provider interfaces.

``BaseMarketDataProvider`` fetches OHLCV for one (symbol, timeframe, window).
Chunking, throttling and checkpointing live in :mod:`trading_ml.data.ingestion`
so the pacing policy is uniform across providers, while each provider only has
to implement a single-window fetch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class BaseMarketDataProvider(ABC):
    """Fetches historical OHLCV bars."""

    #: short provider id, e.g. ``"yfinance"``
    name: str = "base"

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a canonical OHLCV frame for ``[start, end)``.

        Implementations should return a frame normalized via
        :func:`trading_ml.data.schemas.normalize_ohlcv`. May return an empty
        frame when no data is available.
        """

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release any connections. Safe to call multiple times."""

    def __enter__(self) -> BaseMarketDataProvider:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class NewsItem:
    """A single news/event record for the event-driven model (Model 3)."""

    timestamp: pd.Timestamp
    symbol: str
    headline: str
    body: str = ""
    source: str = ""
    url: str = ""


class BaseNewsProvider(ABC):
    """Fetches news items for catalyst detection (Model 3)."""

    name: str = "base-news"

    @abstractmethod
    def fetch_news(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[NewsItem]:
        """Return news items for ``symbol`` within ``[start, end)``."""
