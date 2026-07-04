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


__all__ = [
    "BaseMarketDataProvider",
    "BaseNewsProvider",
    "NewsItem",
    "get_market_provider",
]
