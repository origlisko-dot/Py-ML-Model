"""Sample construction for Model 1.

Turns a symbol's base-timeframe OHLCV into supervised samples. Each sample is a
tensor of shape ``(n_context_timeframes, window, n_features)`` — the last
``window`` *already-closed* bars of every context timeframe as of the decision
time — plus a triple-barrier class label.

Point-in-time (principle #2) is enforced by slicing each timeframe's
close-labeled feature frame with ``searchsorted`` (``side="right"``): only bars
whose close ``<= t`` are visible. Memory (principle #3): the builder consumes one
symbol at a time and callers pass an explicit date range so storage reads only
that slice (DuckDB partition pruning) rather than the whole history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ml.features.indicators import feature_columns
from trading_ml.features.labeling import to_class_index, triple_barrier_labels
from trading_ml.features.multi_timeframe import build_mtf_matrix
from trading_ml.timeframes import sort_labels


@dataclass
class SampleBundle:
    """Numpy sample tensors, torch-free so it can be built and tested anywhere."""

    X: np.ndarray  # (N, n_tf, window, n_feat) float32
    y: np.ndarray  # (N,) int64 in {0,1,2}
    times: np.ndarray  # (N,) datetime64[ns, UTC] decision times
    symbols: np.ndarray  # (N,) object
    feature_names: list[str]
    timeframes: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _clean(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Forward-fill then zero-fill feature columns (warm-up → 0, no leakage)."""
    sub = frame[cols].replace([np.inf, -np.inf], np.nan)
    return sub.ffill().fillna(0.0)


def build_samples(
    base_df: pd.DataFrame,
    symbol: str,
    *,
    base_timeframe: str,
    context_timeframes: list[str],
    window: int,
    horizon: int,
    take_profit_atr: float = 2.0,
    stop_loss_atr: float = 1.0,
    atr_window: int = 14,
    feature_names: list[str] | None = None,
) -> SampleBundle:
    """Build a :class:`SampleBundle` from one symbol's base OHLCV."""
    timeframes = sort_labels(context_timeframes)
    base_feats, per_tf = build_mtf_matrix(base_df, timeframes, base_timeframe, atr_window)

    labels = triple_barrier_labels(base_feats, horizon, take_profit_atr, stop_loss_atr, atr_window)
    labels = labels.dropna(subset=["label"])
    if labels.empty:
        return _empty_bundle(symbol, timeframes, feature_names or [])

    if feature_names is None:
        feature_names = feature_columns(per_tf[timeframes[0]])

    # Pre-extract each timeframe's cleaned feature matrix + its close-time index.
    tf_arrays: dict[str, np.ndarray] = {}
    tf_index: dict[str, pd.DatetimeIndex] = {}
    for tf in timeframes:
        cleaned = _clean(per_tf[tf], feature_names)
        tf_arrays[tf] = cleaned.to_numpy(dtype=np.float32)
        tf_index[tf] = cleaned.index

    xs: list[np.ndarray] = []
    ys: list[int] = []
    ts: list[pd.Timestamp] = []
    y_class = to_class_index(labels["label"])

    for t, cls in zip(labels.index, y_class.to_numpy(), strict=False):
        panes = []
        ok = True
        for tf in timeframes:
            idx = tf_index[tf]
            pos = int(idx.searchsorted(t, side="right"))  # bars with close <= t
            if pos < window:
                ok = False
                break
            panes.append(tf_arrays[tf][pos - window : pos])
        if not ok:
            continue
        xs.append(np.stack(panes, axis=0))
        ys.append(int(cls))
        ts.append(t)

    if not xs:
        return _empty_bundle(symbol, timeframes, feature_names)

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    times = pd.DatetimeIndex(ts).to_numpy()
    symbols = np.array([symbol] * len(ys), dtype=object)
    return SampleBundle(X, y, times, symbols, feature_names, timeframes)


def _empty_bundle(symbol: str, timeframes: list[str], feats: list[str]) -> SampleBundle:
    return SampleBundle(
        X=np.empty((0, len(timeframes), 0, len(feats)), dtype=np.float32),
        y=np.empty((0,), dtype=np.int64),
        times=np.empty((0,), dtype="datetime64[ns]"),
        symbols=np.empty((0,), dtype=object),
        feature_names=feats,
        timeframes=timeframes,
    )


def concat_bundles(bundles: list[SampleBundle]) -> SampleBundle:
    """Concatenate per-symbol bundles into one (sorted by decision time)."""
    bundles = [b for b in bundles if len(b) > 0]
    if not bundles:
        raise ValueError("No samples produced across symbols.")
    ref = bundles[0]
    X = np.concatenate([b.X for b in bundles], axis=0)
    y = np.concatenate([b.y for b in bundles], axis=0)
    times = np.concatenate([b.times for b in bundles], axis=0)
    symbols = np.concatenate([b.symbols for b in bundles], axis=0)
    order = np.argsort(times, kind="stable")
    return SampleBundle(
        X[order], y[order], times[order], symbols[order], ref.feature_names, ref.timeframes
    )


def temporal_split(bundle: SampleBundle, val_fraction: float) -> tuple[SampleBundle, SampleBundle]:
    """Walk-forward split: earliest ``1 - val_fraction`` for train, rest for val.

    No shuffling — prevents the model from validating on samples that precede
    its training data (a subtle leakage).
    """
    n = len(bundle)
    cut = int(n * (1.0 - val_fraction))

    def _slice(a: SampleBundle, lo: int, hi: int) -> SampleBundle:
        return SampleBundle(
            a.X[lo:hi],
            a.y[lo:hi],
            a.times[lo:hi],
            a.symbols[lo:hi],
            a.feature_names,
            a.timeframes,
        )

    return _slice(bundle, 0, cut), _slice(bundle, cut, n)


def fit_scaler(bundle: SampleBundle) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the training bundle (fit on train ONLY)."""
    # X: (N, n_tf, window, n_feat) -> stats over (N, n_tf, window)
    flat = bundle.X.reshape(-1, bundle.n_features)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(bundle: SampleBundle, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((bundle.X - mean) / std).astype(np.float32)
