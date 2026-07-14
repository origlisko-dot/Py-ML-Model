"""Market-data and news provider abstractions and implementations."""

from __future__ import annotations

from trading_ml.data.providers.base import (
    BaseMarketDataProvider,
    BaseNewsProvider,
    NewsItem,
)


def get_market_provider(name: str, **kwargs) -> BaseMarketDataProvider:
    """Factory: resolve a market-data provider by name (lazy imports)."""
    name = name.lower()
    if name in ("yfinance", "yf"):
        from trading_ml.data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider(**kwargs)
    if name in ("ibkr", "ib"):
        from trading_ml.data.providers.ibkr import IBKRProvider

        return IBKRProvider(**kwargs)
    raise ValueError(f"Unknown market provider: {name!r}")


def get_news_provider(name: str, **kwargs) -> BaseNewsProvider:
    """Factory: resolve a news provider by name (lazy imports)."""
    name = name.lower()
    if name in ("yfinance", "yfinance-news", "yf"):
        from trading_ml.data.providers.news_yfinance import YFinanceNewsProvider

        return YFinanceNewsProvider(**kwargs)
    if name in ("memory", "memory-news", "csv"):
        from trading_ml.data.providers.news_memory import InMemoryNewsProvider

        return InMemoryNewsProvider(**kwargs)
    raise ValueError(f"Unknown news provider: {name!r}")


__all__ = [
    "BaseMarketDataProvider",
    "BaseNewsProvider",
    "NewsItem",
    "get_market_provider",
    "get_news_provider",
]
