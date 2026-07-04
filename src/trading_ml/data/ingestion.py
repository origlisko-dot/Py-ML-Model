"""Chunked, throttled, checkpointed OHLCV ingestion.

Why this exists (robustness principle #1 — IBKR pacing): large historical pulls
trip IBKR "pacing violations". Ingestion therefore downloads in small windows
(``chunk``), **writes each chunk immediately** (checkpoint) and **throttles**
between requests. If the run dies, ``resume=True`` continues from the last
stored bar (``storage.available_range``) instead of restarting.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from trading_ml.config import load_config
from trading_ml.data import storage
from trading_ml.data.providers import get_market_provider
from trading_ml.data.providers.base import BaseMarketDataProvider
from trading_ml.timeframes import duration_to_timedelta


def _parse_window(spec: str) -> pd.Timedelta:
    """Parse a window spec like ``30d`` / ``1W`` (case-insensitive unit)."""
    m = re.match(r"^(\d+)([mhdwMHDW])$", spec)
    if not m:
        raise ValueError(f"Invalid window spec: {spec!r}")
    unit = m.group(2).lower()
    return duration_to_timedelta(f"{m.group(1)}{unit}")


@dataclass
class IngestReport:
    symbol: str
    timeframe: str
    rows: int = 0
    chunks: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    errors: list[str] = field(default_factory=list)


def ingest_series(
    provider: BaseMarketDataProvider,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk: str = "1W",
    throttle_seconds: float = 0.0,
    max_retries: int = 3,
    resume: bool = True,
) -> IngestReport:
    """Ingest one (symbol, timeframe) over ``[start, end)`` with checkpointing."""
    report = IngestReport(symbol=symbol, timeframe=timeframe, start=start, end=end)
    chunk_td = _parse_window(chunk)

    if resume:
        rng = storage.available_range(symbol, timeframe)
        if rng is not None and rng[1] >= start:
            # Resume just after the last stored bar.
            start = rng[1] + pd.Timedelta(seconds=1)

    cursor = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")

    while cursor < end:
        chunk_end = min(cursor + chunk_td, end)
        df = pd.DataFrame()
        for attempt in range(1, max_retries + 1):
            try:
                df = provider.fetch_ohlcv(symbol, timeframe, cursor, chunk_end)
                break
            except Exception as exc:  # noqa: BLE001 - record & backoff
                report.errors.append(f"{cursor.date()}..{chunk_end.date()}: {exc}")
                if attempt == max_retries:
                    df = pd.DataFrame()
                else:
                    time.sleep(min(2.0 * attempt, 8.0))

        if df is not None and not df.empty:
            # Checkpoint: persist this chunk before moving on.
            report.rows += storage.write_ohlcv(symbol, timeframe, df)
        report.chunks += 1

        cursor = chunk_end
        if throttle_seconds > 0 and cursor < end:
            time.sleep(throttle_seconds)

    return report


def ingest(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    provider_name: str | None = None,
    end: pd.Timestamp | None = None,
    resume: bool = True,
) -> list[IngestReport]:
    """Ingest all configured (symbol, timeframe) pairs.

    History windows per timeframe come from ``config/data.yaml``. Returns one
    :class:`IngestReport` per series.
    """
    cfg = load_config()
    symbols = symbols or cfg.data.symbols
    timeframes = timeframes or cfg.data.timeframes
    provider_name = provider_name or cfg.data.provider
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    provider = get_market_provider(
        provider_name,
        min_spacing_seconds=cfg.data.ibkr.throttle_seconds,
    )
    reports: list[IngestReport] = []
    try:
        for symbol in symbols:
            for tf in timeframes:
                window = cfg.data.history.get(tf, "365d")
                start = end - _parse_window(window)
                reports.append(
                    ingest_series(
                        provider,
                        symbol,
                        tf,
                        start,
                        end,
                        chunk=cfg.data.ibkr.chunk,
                        throttle_seconds=(
                            cfg.data.ibkr.throttle_seconds
                            if provider_name.lower().startswith("ib")
                            else 0.0
                        ),
                        max_retries=cfg.data.ibkr.max_retries,
                        resume=resume,
                    )
                )
    finally:
        provider.close()
    return reports
