"""Robustness principle #1: chunked ingestion with checkpoint-based resume."""

import numpy as np
import pandas as pd

from trading_ml.data import storage
from trading_ml.data.ingestion import ingest_series
from trading_ml.data.providers.base import BaseMarketDataProvider


class FakeProvider(BaseMarketDataProvider):
    """Deterministic minute-bar generator that records requested windows."""

    name = "fake"

    def __init__(self):
        self.requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch_ohlcv(self, symbol, timeframe, start, end):
        self.requests.append((pd.Timestamp(start), pd.Timestamp(end)))
        idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
        if len(idx) == 0:
            from trading_ml.data.schemas import empty_ohlcv

            return empty_ohlcv()
        n = len(idx)
        price = 100 + np.arange(n) * 0.01
        return pd.DataFrame(
            {"open": price, "high": price + 0.1, "low": price - 0.1, "close": price, "volume": 1.0},
            index=pd.DatetimeIndex(idx, name="timestamp"),
        )


def _patch_root(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_root", lambda: tmp_path)


def test_chunked_download_covers_range(monkeypatch, tmp_path):
    _patch_root(monkeypatch, tmp_path)
    provider = FakeProvider()
    start = pd.Timestamp("2024-06-03", tz="UTC")
    end = pd.Timestamp("2024-06-06", tz="UTC")

    report = ingest_series(provider, "SYM", "1m", start, end, chunk="1D", throttle_seconds=0.0)

    assert report.rows > 0
    # 3-day span in 1D chunks -> 3 requests.
    assert report.chunks == 3
    stored = storage.read_ohlcv("SYM", "1m", root=tmp_path)
    assert len(stored) > 0


def test_resume_from_checkpoint(monkeypatch, tmp_path):
    _patch_root(monkeypatch, tmp_path)
    start = pd.Timestamp("2024-06-03", tz="UTC")
    mid = pd.Timestamp("2024-06-05", tz="UTC")
    end = pd.Timestamp("2024-06-07", tz="UTC")

    # First pass: ingest only the first half.
    p1 = FakeProvider()
    ingest_series(p1, "SYM", "1m", start, mid, chunk="1D", throttle_seconds=0.0)
    checkpoint = storage.available_range("SYM", "1m", root=tmp_path)[1]

    # Second pass over the FULL range with resume=True must NOT re-request
    # anything before the checkpoint.
    p2 = FakeProvider()
    ingest_series(p2, "SYM", "1m", start, end, chunk="1D", throttle_seconds=0.0, resume=True)

    assert p2.requests, "resume pass should still fetch the remaining window"
    earliest_requested = min(r[0] for r in p2.requests)
    assert earliest_requested >= checkpoint

    # Final store spans the whole range.
    rng = storage.available_range("SYM", "1m", root=tmp_path)
    assert rng[1] >= pd.Timestamp("2024-06-06", tz="UTC")


def test_retry_on_transient_error(monkeypatch, tmp_path):
    _patch_root(monkeypatch, tmp_path)

    class FlakyProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def fetch_ohlcv(self, symbol, timeframe, start, end):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return super().fetch_ohlcv(symbol, timeframe, start, end)

    provider = FlakyProvider()
    start = pd.Timestamp("2024-06-03", tz="UTC")
    end = pd.Timestamp("2024-06-04", tz="UTC")
    report = ingest_series(
        provider, "SYM", "1m", start, end, chunk="1D", throttle_seconds=0.0, max_retries=3
    )
    assert report.rows > 0  # recovered after retry
