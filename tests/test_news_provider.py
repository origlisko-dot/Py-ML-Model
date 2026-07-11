"""News providers: in-memory slicing, factory resolution, yfinance parsing."""

import pandas as pd
import pytest

from trading_ml.data.providers import get_news_provider
from trading_ml.data.providers.base import NewsItem
from trading_ml.data.providers.news_memory import InMemoryNewsProvider
from trading_ml.data.providers.news_yfinance import YFinanceNewsProvider


def _item(ts: str, symbol: str = "AAPL", headline: str = "h") -> NewsItem:
    return NewsItem(timestamp=pd.Timestamp(ts, tz="UTC"), symbol=symbol, headline=headline)


def test_memory_provider_filters_by_range_and_symbol():
    items = [
        _item("2024-06-01 12:00", "AAPL", "a"),
        _item("2024-06-03 12:00", "AAPL", "b"),
        _item("2024-06-05 12:00", "AAPL", "c"),
        _item("2024-06-03 12:00", "MSFT", "d"),
    ]
    prov = InMemoryNewsProvider(items)
    out = prov.fetch_news(
        "AAPL", pd.Timestamp("2024-06-02", tz="UTC"), pd.Timestamp("2024-06-04", tz="UTC")
    )
    assert [it.headline for it in out] == ["b"]  # in-range AAPL only


def test_memory_provider_end_is_exclusive():
    prov = InMemoryNewsProvider([_item("2024-06-04 00:00", headline="edge")])
    out = prov.fetch_news(
        "AAPL", pd.Timestamp("2024-06-03", tz="UTC"), pd.Timestamp("2024-06-04", tz="UTC")
    )
    assert out == []  # end is exclusive


def test_factory_resolves_providers():
    assert get_news_provider("memory").name == "memory-news"
    assert isinstance(get_news_provider("csv"), InMemoryNewsProvider)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown news provider"):
        get_news_provider("nope")


def test_yfinance_parses_flat_payload():
    entry = {
        "title": "Acme to acquire Beta",
        "publisher": "Reuters",
        "link": "http://x/y",
        "providerPublishTime": int(pd.Timestamp("2024-06-03 12:00", tz="UTC").timestamp()),
    }
    item = YFinanceNewsProvider._to_item(entry, "AAPL")
    assert item is not None
    assert item.headline == "Acme to acquire Beta"
    assert item.source == "Reuters"
    assert item.url == "http://x/y"
    assert item.timestamp == pd.Timestamp("2024-06-03 12:00", tz="UTC")


def test_yfinance_parses_nested_payload():
    entry = {
        "content": {
            "title": "Big partnership announced",
            "pubDate": "2024-06-03T12:00:00Z",
            "provider": {"displayName": "Bloomberg"},
            "canonicalUrl": {"url": "http://a/b"},
            "summary": "details",
        }
    }
    item = YFinanceNewsProvider._to_item(entry, "MSFT")
    assert item is not None
    assert item.headline == "Big partnership announced"
    assert item.source == "Bloomberg"
    assert item.url == "http://a/b"
    assert item.body == "details"


def test_yfinance_skips_untitled_or_undated():
    assert YFinanceNewsProvider._to_item({"providerPublishTime": 1}, "AAPL") is None
    assert YFinanceNewsProvider._to_item({"title": "no date"}, "AAPL") is None


def test_yfinance_fetch_filters_range(monkeypatch):
    payload = [
        {
            "title": "old",
            "providerPublishTime": int(pd.Timestamp("2024-05-01", tz="UTC").timestamp()),
        },
        {
            "title": "mid",
            "providerPublishTime": int(pd.Timestamp("2024-06-03", tz="UTC").timestamp()),
        },
        {
            "title": "new",
            "providerPublishTime": int(pd.Timestamp("2024-07-01", tz="UTC").timestamp()),
        },
    ]

    class FakeTicker:
        def __init__(self, symbol):
            self.news = payload

    fake_yf = type("yf", (), {"Ticker": FakeTicker})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    prov = YFinanceNewsProvider.__new__(YFinanceNewsProvider)  # skip import check
    out = prov.fetch_news(
        "AAPL", pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-06-30", tz="UTC")
    )
    assert [it.headline for it in out] == ["mid"]
