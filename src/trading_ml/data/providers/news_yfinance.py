"""yfinance news provider — the default news source for development and CI.

Reads headlines from ``yfinance.Ticker(symbol).news``. yfinance has shipped two
payload shapes over time: a flat one (``title``/``providerPublishTime``/``link``)
and a nested one (``{"content": {...}}``). :meth:`_to_item` normalizes both.

Yahoo's news feed only exposes recent items and gives no server-side time
window, so :meth:`fetch_news` filters client-side to ``[start, end)``.
"""

from __future__ import annotations

import pandas as pd

from trading_ml.data.providers.base import BaseNewsProvider, NewsItem


class YFinanceNewsProvider(BaseNewsProvider):
    name = "yfinance-news"

    def __init__(self, **_: object) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "yfinance is required. Install with `uv sync --extra providers`."
            ) from exc

    def fetch_news(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[NewsItem]:
        import yfinance as yf

        raw = getattr(yf.Ticker(symbol), "news", None) or []
        start = pd.Timestamp(start).tz_convert("UTC")
        end = pd.Timestamp(end).tz_convert("UTC")

        items: list[NewsItem] = []
        for entry in raw:
            item = self._to_item(entry, symbol)
            if item is None:
                continue
            if start <= item.timestamp < end:
                items.append(item)
        items.sort(key=lambda it: it.timestamp)
        return items

    @staticmethod
    def _to_item(entry: dict, symbol: str) -> NewsItem | None:
        """Normalize one yfinance news dict (flat or nested) to a NewsItem."""
        content = entry.get("content", entry)  # nested shape wraps fields in `content`

        headline = content.get("title") or entry.get("title") or ""
        if not headline:
            return None

        ts = _parse_timestamp(entry, content)
        if ts is None:
            return None

        summary = content.get("summary") or content.get("description") or ""
        source = _extract_source(entry, content)
        url = _extract_url(entry, content)

        return NewsItem(
            timestamp=ts,
            symbol=symbol,
            headline=str(headline),
            body=str(summary),
            source=str(source),
            url=str(url),
        )


def _parse_timestamp(entry: dict, content: dict) -> pd.Timestamp | None:
    """Resolve a UTC timestamp from either payload shape, or None if absent."""
    epoch = entry.get("providerPublishTime")
    if epoch is not None:
        return pd.Timestamp(int(epoch), unit="s", tz="UTC")
    iso = content.get("pubDate") or content.get("displayTime")
    if iso:
        try:
            return pd.Timestamp(iso).tz_convert("UTC")
        except (ValueError, TypeError):
            return None
    return None


def _extract_source(entry: dict, content: dict) -> str:
    provider = content.get("provider")
    if isinstance(provider, dict):
        return provider.get("displayName", "") or ""
    return entry.get("publisher", "") or ""


def _extract_url(entry: dict, content: dict) -> str:
    url = content.get("canonicalUrl") or content.get("clickThroughUrl")
    if isinstance(url, dict):
        return url.get("url", "") or ""
    return entry.get("link", "") or ""
