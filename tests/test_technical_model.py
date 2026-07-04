"""Model 1 dataset + network. Skipped when the ML extra (torch) is absent."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from trading_ml.models.technical.dataset import (  # noqa: E402
    build_samples,
    fit_scaler,
    temporal_split,
)


def _many_days(n_days=40, start="2024-04-01", seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    day = pd.Timestamp(start, tz="UTC")
    price = 100.0
    made = 0
    while made < n_days:
        if day.weekday() < 5:
            idx = pd.date_range(day + pd.Timedelta(hours=14), periods=78, freq="5min")
            close = price + np.cumsum(rng.normal(0, 0.15, 78))
            o = close - rng.normal(0, 0.05, 78)
            h = np.maximum(o, close) + np.abs(rng.normal(0, 0.08, 78))
            low = np.minimum(o, close) - np.abs(rng.normal(0, 0.08, 78))
            frames.append(
                pd.DataFrame(
                    {"open": o, "high": h, "low": low, "close": close, "volume": 1.0}, index=idx
                )
            )
            price = close[-1]
            made += 1
        day += pd.Timedelta(days=1)
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df


@pytest.fixture(scope="module")
def base_df():
    return _many_days()


def test_build_samples_shape(base_df):
    bundle = build_samples(
        base_df,
        "SYNTH",
        base_timeframe="5m",
        context_timeframes=["5m", "15m", "1h"],
        window=16,
        horizon=6,
    )
    assert len(bundle) > 0
    # (N, n_tf, window, n_feat)
    assert bundle.X.shape[1] == 3
    assert bundle.X.shape[2] == 16
    assert bundle.X.shape[3] == bundle.n_features
    assert set(np.unique(bundle.y)).issubset({0, 1, 2})
    assert np.isfinite(bundle.X).all()


def test_temporal_split_is_ordered(base_df):
    bundle = build_samples(
        base_df,
        "S",
        base_timeframe="5m",
        context_timeframes=["5m", "15m"],
        window=16,
        horizon=6,
    )
    train, val = temporal_split(bundle, 0.2)
    assert len(train) + len(val) == len(bundle)
    # Walk-forward: all training times precede all validation times.
    assert train.times.max() <= val.times.min()


def test_scaler_fit_on_train_only(base_df):
    bundle = build_samples(
        base_df,
        "S",
        base_timeframe="5m",
        context_timeframes=["5m", "15m"],
        window=16,
        horizon=6,
    )
    train, _ = temporal_split(bundle, 0.2)
    mean, std = fit_scaler(train)
    assert mean.shape == (bundle.n_features,)
    assert (std > 0).all()


def test_tiny_fit_predict(base_df):
    cfg = {
        "data": {
            "base_timeframe": "5m",
            "context_timeframes": ["5m", "15m"],
            "window": 16,
            "horizon": 6,
        },
        "labeling": {"take_profit_atr": 1.5, "stop_loss_atr": 1.0, "atr_window": 14},
        "network": {
            "encoder": "tcn",
            "d_model": 16,
            "n_heads": 2,
            "encoder_layers": 1,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 64,
            "max_epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "val_fraction": 0.2,
            "early_stopping_patience": 2,
            "num_workers": 0,
        },
    }
    from trading_ml.models.technical.module import TechnicalModel

    model = TechnicalModel(cfg).fit({"S": base_df})
    sigs = model.predict({"S": base_df})
    assert len(sigs) > 0
    assert all(s.direction in (-1, 0, 1) for s in sigs)
    assert all(0.0 <= s.probability <= 1.0 for s in sigs)
