"""News ingestion: chunked fetch, checkpoint-based resume, retry on error."""

import pandas as pd

from trading_ml.data import news_storage
from trading_ml.data.news_ingestion import ingest_news_series
from trading_ml.data.providers.base import BaseNewsProvider, NewsItem


class FakeNewsProvider(BaseNewsProvider):
    """One item per requested day; records the requested windows."""

    name = "fake-news"

    def __init__(self):
        self.requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch_news(self, symbol, start, end):
        self.requests.append((pd.Timestamp(start), pd.Timestamp(end)))
        days = pd.date_range(start, end, freq="1D", inclusive="left", tz="UTC")
        return [NewsItem(ts, symbol, f"headline {ts.date()}") for ts in days]


def _patch_root(monkeypatch, tmp_path):
    monkeypatch.setattr(news_storage, "_root", lambda: tmp_path)


def test_chunked_fetch_covers_range(monkeypatch, tmp_path):
    _patch_root(monkeypatch, tmp_path)
    provider = FakeNewsProvider()
    start = pd.Timestamp("2024-06-03", tz="UTC")
    end = pd.Timestamp("2024-06-06", tz="UTC")

    report = ingest_news_series(provider, "SYM", start, end, chunk="1d", throttle_seconds=0.0)

    assert report.items == 3  # one per day
    assert report.chunks == 3
    stored = news_storage.read_news("SYM", root=tmp_path)
    assert len(stored) == 3


def test_resume_from_checkpoint(monkeypatch, tmp_path):
    _patch_root(monkeypatch, tmp_path)
    start = pd.Timestamp("2024-06-03", tz="UTC")
    mid = pd.Timestamp("2024-06-05", tz="UTC")
    end = pd.Timestamp("2024-06-07", tz="UTC")

    ingest_news_series(FakeNewsProvider(), "SYM", start, mid, chunk="1d", throttle_seconds=0.0)
    checkpoint = news_storage.available_news_range("SYM", root=tmp_path)[1]

    p2 = FakeNewsProvider()
    ingest_news_series(p2, "SYM", start, end, chunk="1d", throttle_seconds=0.0, resume=True)

    assert p2.requests, "resume pass should still fetch the remaining window"
    earliest = min(r[0] for r in p2.requests)
    assert earliest >= checkpoint  # nothing before the checkpoint refetched


def test_retry_on_transient_error(monkeypatch, tmp_path):
    _patch_root(monkeypatch, tmp_path)

    class FlakyProvider(FakeNewsProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def fetch_news(self, symbol, start, end):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return super().fetch_news(symbol, start, end)

    provider = FlakyProvider()
    start = pd.Timestamp("2024-06-03", tz="UTC")
    end = pd.Timestamp("2024-06-04", tz="UTC")
    report = ingest_news_series(
        provider, "SYM", start, end, chunk="1d", throttle_seconds=0.0, max_retries=3
    )
    assert report.items == 1  # recovered after retry
