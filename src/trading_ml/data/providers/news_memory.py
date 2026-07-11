"""In-memory / CSV news provider — deterministic source for tests and CI.

Holds a fixed list of :class:`NewsItem` (or loads one from a CSV) and serves
the ``[start, end)`` slice. Requires no network, so it exercises the whole
news pathway — ingestion, storage, classification, fusion — reproducibly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_ml.data.providers.base import BaseNewsProvider, NewsItem


class InMemoryNewsProvider(BaseNewsProvider):
    name = "memory-news"

    def __init__(self, items: list[NewsItem] | None = None, **_: object) -> None:
        self._items = sorted(items or [], key=lambda it: it.timestamp)

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: object) -> InMemoryNewsProvider:
        """Build from a CSV with columns: timestamp, symbol, headline[, body, source, url]."""
        df = pd.read_csv(path)
        items = [
            NewsItem(
                timestamp=pd.Timestamp(row["timestamp"]).tz_convert("UTC")
                if pd.Timestamp(row["timestamp"]).tzinfo
                else pd.Timestamp(row["timestamp"]).tz_localize("UTC"),
                symbol=str(row["symbol"]),
                headline=str(row["headline"]),
                body=str(row.get("body", "") or ""),
                source=str(row.get("source", "") or ""),
                url=str(row.get("url", "") or ""),
            )
            for _, row in df.iterrows()
        ]
        return cls(items, **kwargs)

    def fetch_news(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[NewsItem]:
        start = pd.Timestamp(start).tz_convert("UTC")
        end = pd.Timestamp(end).tz_convert("UTC")
        return [it for it in self._items if it.symbol == symbol and start <= it.timestamp < end]
