"""Partitioned Parquet storage for news items (Model 3).

Mirrors :mod:`trading_ml.data.storage` (OHLCV) but for :class:`NewsItem`
records. Layout::

    {parquet_dir}/news/symbol={SYM}/year={YYYY}/month={MM}/data.parquet

Writes merge into the target month partition and de-duplicate on
``(timestamp, headline)`` — the same story can arrive from multiple fetches.
Reads go through DuckDB with ``year``/``month`` pruning and optional time-range
filters, returning only the requested slice.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from trading_ml.config import load_config
from trading_ml.data.providers.base import NewsItem

_COLUMNS = ["timestamp", "symbol", "headline", "body", "source", "url"]


def _root() -> Path:
    return Path(load_config().paths.parquet_dir) / "news"


def _partition_dir(root: Path, symbol: str, year: int, month: int) -> Path:
    return root / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}"


def _series_dir(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}"


def items_to_frame(items: list[NewsItem]) -> pd.DataFrame:
    """Convert NewsItems to a canonical frame (UTC timestamp column, sorted)."""
    if not items:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _COLUMNS})
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(it.timestamp).tz_convert("UTC") for it in items],
            "symbol": [it.symbol for it in items],
            "headline": [it.headline for it in items],
            "body": [it.body for it in items],
            "source": [it.source for it in items],
            "url": [it.url for it in items],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def frame_to_items(df: pd.DataFrame) -> list[NewsItem]:
    """Convert a stored frame back to NewsItems."""
    return [
        NewsItem(
            timestamp=pd.Timestamp(row["timestamp"]),
            symbol=str(row["symbol"]),
            headline=str(row["headline"]),
            body=str(row.get("body", "") or ""),
            source=str(row.get("source", "") or ""),
            url=str(row.get("url", "") or ""),
        )
        for _, row in df.iterrows()
    ]


def write_news(symbol: str, items: list[NewsItem], root: Path | None = None) -> int:
    """Write/merge news items into the partitioned store.

    Returns the number of rows written (post-merge, this partition set).
    Existing rows with the same ``(timestamp, headline)`` are kept once.
    """
    root = root or _root()
    df = items_to_frame(items)
    if df.empty:
        return 0

    written = 0
    keys = pd.DataFrame({"year": df["timestamp"].dt.year, "month": df["timestamp"].dt.month})
    for (year, month), idx in keys.groupby(["year", "month"]).groups.items():
        part = df.loc[idx]
        pdir = _partition_dir(root, symbol, int(year), int(month))
        pdir.mkdir(parents=True, exist_ok=True)
        fpath = pdir / "data.parquet"

        if fpath.exists():
            existing = pd.read_parquet(fpath)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            part = pd.concat([existing, part], ignore_index=True)

        part = (
            part.drop_duplicates(subset=["timestamp", "headline"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        part.to_parquet(fpath, index=False)
        written += len(part)
    return written


def read_news(
    symbol: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    root: Path | None = None,
) -> list[NewsItem]:
    """Read a (sliced) list of news items via DuckDB with partition pruning."""
    root = root or _root()
    sdir = _series_dir(root, symbol)
    if not sdir.exists():
        return []

    glob = str(sdir / "**" / "*.parquet")
    source = f"read_parquet('{glob}', hive_partitioning=1, union_by_name=1)"
    query = f"SELECT timestamp, symbol, headline, body, source, url FROM {source}"
    preds = []
    if start is not None:
        preds.append(f"timestamp >= TIMESTAMPTZ '{pd.Timestamp(start).isoformat()}'")
    if end is not None:
        preds.append(f"timestamp < TIMESTAMPTZ '{pd.Timestamp(end).isoformat()}'")
    if preds:
        query += " WHERE " + " AND ".join(preds)
    query += " ORDER BY timestamp"

    con = duckdb.connect(database=":memory:")
    try:
        df = con.sql(query).df()
    finally:
        con.close()

    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return frame_to_items(df)


def available_news_range(
    symbol: str, root: Path | None = None
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return ``(min_ts, max_ts)`` stored for a symbol, or ``None`` if empty.

    Used for checkpointing: ingestion resumes after ``max_ts``.
    """
    root = root or _root()
    sdir = _series_dir(root, symbol)
    if not sdir.exists():
        return None
    glob = str(sdir / "**" / "*.parquet")
    con = duckdb.connect(database=":memory:")
    try:
        row = con.sql(
            f"SELECT min(timestamp) AS lo, max(timestamp) AS hi "
            f"FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=1)"
        ).fetchone()
    except duckdb.Error:
        return None
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    return pd.Timestamp(row[0]).tz_convert("UTC"), pd.Timestamp(row[1]).tz_convert("UTC")
