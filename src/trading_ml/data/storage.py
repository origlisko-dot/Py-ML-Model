"""Partitioned Parquet storage with DuckDB-backed reads.

Layout (Hive-style partitions keep minute data manageable and enable
predicate pushdown so reads never load the whole history into RAM):

    {parquet_dir}/symbol={SYM}/timeframe={TF}/year={YYYY}/month={MM}/data.parquet

Writes merge into the target month partition (dedup on timestamp). Reads go
through DuckDB with ``year``/``month`` pruning and optional time-range filters,
returning only the requested slice — the basis for the model's lazy loading.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from trading_ml.config import load_config
from trading_ml.data.schemas import empty_ohlcv, normalize_ohlcv


def _root() -> Path:
    return Path(load_config().paths.parquet_dir)


def _partition_dir(root: Path, symbol: str, timeframe: str, year: int, month: int) -> Path:
    return (
        root / f"symbol={symbol}" / f"timeframe={timeframe}" / f"year={year}" / f"month={month:02d}"
    )


def _series_dir(root: Path, symbol: str, timeframe: str) -> Path:
    return root / f"symbol={symbol}" / f"timeframe={timeframe}"


def write_ohlcv(symbol: str, timeframe: str, df: pd.DataFrame, root: Path | None = None) -> int:
    """Write/merge an OHLCV frame into the partitioned store.

    Returns the number of rows written (post-merge, this partition set).
    Existing rows with the same timestamp are overwritten by the new data.
    """
    root = root or _root()
    df = normalize_ohlcv(df)
    if df.empty:
        return 0

    written = 0
    keys = pd.DataFrame({"year": df.index.year, "month": df.index.month}, index=df.index)
    for (year, month), idx in keys.groupby(["year", "month"]).groups.items():
        part = df.loc[idx]
        pdir = _partition_dir(root, symbol, timeframe, int(year), int(month))
        pdir.mkdir(parents=True, exist_ok=True)
        fpath = pdir / "data.parquet"

        if fpath.exists():
            existing = pd.read_parquet(fpath)
            existing.index = pd.to_datetime(existing.index, utc=True)
            existing.index.name = "timestamp"
            part = pd.concat([existing, part])
            part = part[~part.index.duplicated(keep="last")].sort_index()

        part.to_parquet(fpath, index=True)
        written += len(part)
    return written


def read_ohlcv(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    columns: list[str] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Read a (sliced) OHLCV frame via DuckDB with partition pruning.

    Only the requested time range is materialized — safe for large histories.
    """
    root = root or _root()
    sdir = _series_dir(root, symbol, timeframe)
    if not sdir.exists():
        return empty_ohlcv()

    glob = str(sdir / "**" / "*.parquet")
    cols = "*" if not columns else ", ".join({"timestamp", *columns})
    source = f"read_parquet('{glob}', hive_partitioning=1, union_by_name=1)"
    query = f"SELECT {cols} FROM {source}"
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
        return empty_ohlcv()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Drop hive partition helper columns if present.
    df = df.drop(columns=[c for c in ("year", "month") if c in df.columns])
    return df


def available_range(
    symbol: str, timeframe: str, root: Path | None = None
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return ``(min_ts, max_ts)`` stored for a series, or ``None`` if empty.

    Used for checkpointing: ingestion resumes after ``max_ts``.
    """
    root = root or _root()
    sdir = _series_dir(root, symbol, timeframe)
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


def list_symbols(root: Path | None = None) -> list[str]:
    root = root or _root()
    if not root.exists():
        return []
    return sorted(p.name.split("=", 1)[1] for p in root.glob("symbol=*") if p.is_dir())
