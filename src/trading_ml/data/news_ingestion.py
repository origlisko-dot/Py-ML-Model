"""Chunked, throttled, checkpointed news ingestion (Model 3).

The news analogue of :mod:`trading_ml.data.ingestion`. News feeds are lighter
than OHLCV, but the same robustness policy applies: fetch in windows, **write
each chunk immediately** (checkpoint) and **throttle** between requests. With
``resume=True`` a re-run continues after the last stored item
(:func:`news_storage.available_news_range`) instead of refetching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from trading_ml.config import load_config
from trading_ml.data import news_storage
from trading_ml.data.ingestion import _parse_window
from trading_ml.data.providers import get_news_provider
from trading_ml.data.providers.base import BaseNewsProvider


@dataclass
class NewsIngestReport:
    symbol: str
    items: int = 0
    chunks: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    errors: list[str] = field(default_factory=list)


def ingest_news_series(
    provider: BaseNewsProvider,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk: str = "7d",
    throttle_seconds: float = 0.0,
    max_retries: int = 3,
    resume: bool = True,
) -> NewsIngestReport:
    """Ingest news for one symbol over ``[start, end)`` with checkpointing."""
    report = NewsIngestReport(symbol=symbol, start=start, end=end)
    chunk_td = _parse_window(chunk)

    if resume:
        rng = news_storage.available_news_range(symbol)
        if rng is not None and rng[1] >= start:
            start = rng[1] + pd.Timedelta(seconds=1)

    cursor = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")

    while cursor < end:
        chunk_end = min(cursor + chunk_td, end)
        items = []
        for attempt in range(1, max_retries + 1):
            try:
                items = provider.fetch_news(symbol, cursor, chunk_end)
                break
            except Exception as exc:  # noqa: BLE001 - record & backoff
                report.errors.append(f"{cursor.date()}..{chunk_end.date()}: {exc}")
                if attempt == max_retries:
                    items = []
                else:
                    time.sleep(min(2.0 * attempt, 8.0))

        if items:
            # Checkpoint: persist this chunk before moving on.
            news_storage.write_news(symbol, items)
            report.items += len(items)
        report.chunks += 1

        cursor = chunk_end
        if throttle_seconds > 0 and cursor < end:
            time.sleep(throttle_seconds)

    return report


def ingest_news(
    symbols: list[str] | None = None,
    provider_name: str | None = None,
    lookback: str = "30d",
    end: pd.Timestamp | None = None,
    resume: bool = True,
    **provider_kwargs: object,
) -> list[NewsIngestReport]:
    """Ingest news for all configured symbols.

    The active news provider comes from ``config/data.yaml`` (``news_provider``).
    Returns one :class:`NewsIngestReport` per symbol.
    """
    cfg = load_config()
    symbols = symbols or cfg.data.symbols
    provider_name = provider_name or cfg.data.news_provider
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    start = end - _parse_window(lookback)

    provider = get_news_provider(provider_name, **provider_kwargs)
    reports: list[NewsIngestReport] = []
    for symbol in symbols:
        reports.append(
            ingest_news_series(
                provider,
                symbol,
                start,
                end,
                chunk=cfg.data.news.chunk,
                throttle_seconds=cfg.data.news.throttle_seconds,
                max_retries=cfg.data.news.max_retries,
                resume=resume,
            )
        )
    return reports
