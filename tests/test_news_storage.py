"""News storage round-trip: partitioned layout, range reads, dedup, checkpoint."""

import pandas as pd

from trading_ml.data import news_storage
from trading_ml.data.providers.base import NewsItem


def _items(symbol="AAPL"):
    return [
        NewsItem(pd.Timestamp("2024-06-01 12:00", tz="UTC"), symbol, "acquire deal", source="R"),
        NewsItem(pd.Timestamp("2024-06-15 12:00", tz="UTC"), symbol, "partnership", source="B"),
        NewsItem(pd.Timestamp("2024-07-02 12:00", tz="UTC"), symbol, "earnings beat", source="R"),
    ]


def test_write_read_roundtrip(tmp_path):
    n = news_storage.write_news("AAPL", _items(), root=tmp_path)
    assert n == 3  # 2 items in the June partition + 1 in July
    back = news_storage.read_news("AAPL", root=tmp_path)
    assert [it.headline for it in back] == ["acquire deal", "partnership", "earnings beat"]
    assert back[0].source == "R"


def test_partitioned_layout_and_range_read(tmp_path):
    news_storage.write_news("AAPL", _items(), root=tmp_path)
    assert (tmp_path / "symbol=AAPL" / "year=2024" / "month=06").exists()
    assert (tmp_path / "symbol=AAPL" / "year=2024" / "month=07").exists()

    sliced = news_storage.read_news(
        "AAPL", start=pd.Timestamp("2024-07-01", tz="UTC"), root=tmp_path
    )
    assert [it.headline for it in sliced] == ["earnings beat"]


def test_merge_dedup_on_rewrite(tmp_path):
    news_storage.write_news("AAPL", _items(), root=tmp_path)
    news_storage.write_news("AAPL", _items(), root=tmp_path)  # same items again
    back = news_storage.read_news("AAPL", root=tmp_path)
    assert len(back) == 3  # no duplicates on (timestamp, headline)


def test_available_range_checkpoint(tmp_path):
    news_storage.write_news("AAPL", _items(), root=tmp_path)
    rng = news_storage.available_news_range("AAPL", root=tmp_path)
    assert rng is not None
    assert rng[0] == pd.Timestamp("2024-06-01 12:00", tz="UTC")
    assert rng[1] == pd.Timestamp("2024-07-02 12:00", tz="UTC")


def test_available_range_none_when_empty(tmp_path):
    assert news_storage.available_news_range("NOPE", root=tmp_path) is None
    assert news_storage.read_news("NOPE", root=tmp_path) == []
