"""Storage round-trip and robustness principle #3: partition-pruned lazy reads."""

from trading_ml.data import storage
from trading_ml.features.multi_timeframe import resample_ohlcv


def test_write_read_roundtrip(tmp_path, minute_ohlcv):
    five = resample_ohlcv(minute_ohlcv, "5m")
    n = storage.write_ohlcv("TEST", "5m", five, root=tmp_path)
    assert n == len(five)
    back = storage.read_ohlcv("TEST", "5m", root=tmp_path)
    assert len(back) == len(five)
    assert list(back.columns[:5]) == ["open", "high", "low", "close", "volume"]


def test_available_range_checkpoint(tmp_path, minute_ohlcv):
    five = resample_ohlcv(minute_ohlcv, "5m")
    storage.write_ohlcv("TEST", "5m", five, root=tmp_path)
    rng = storage.available_range("TEST", "5m", root=tmp_path)
    assert rng is not None
    assert rng[0] == five.index.min()
    assert rng[1] == five.index.max()


def test_partitioned_layout_and_range_read(tmp_path, minute_ohlcv):
    five = resample_ohlcv(minute_ohlcv, "5m")
    storage.write_ohlcv("TEST", "5m", five, root=tmp_path)
    # Hive-partitioned directories exist (symbol/timeframe/year/month).
    assert (tmp_path / "symbol=TEST" / "timeframe=5m").exists()
    assert list((tmp_path / "symbol=TEST" / "timeframe=5m").glob("year=*/month=*/*.parquet"))

    # A range read returns only the requested slice (predicate pushdown), which
    # is strictly smaller than the full store — the basis for lazy loading.
    mid = five.index[len(five) // 2]
    sliced = storage.read_ohlcv("TEST", "5m", start=mid, root=tmp_path)
    assert len(sliced) < len(five)
    assert sliced.index.min() >= mid


def test_merge_dedup_on_rewrite(tmp_path, minute_ohlcv):
    five = resample_ohlcv(minute_ohlcv, "5m")
    storage.write_ohlcv("TEST", "5m", five, root=tmp_path)
    storage.write_ohlcv("TEST", "5m", five, root=tmp_path)  # rewrite same data
    back = storage.read_ohlcv("TEST", "5m", root=tmp_path)
    assert len(back) == len(five)  # no duplicates
    assert not back.index.duplicated().any()
