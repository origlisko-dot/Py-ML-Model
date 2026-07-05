"""Walk-forward CV: splits (NumPy) and validation (torch-gated)."""

import numpy as np
import pytest

from trading_ml.models.technical.walkforward import walk_forward_splits


def test_splits_are_chronological_and_non_overlapping():
    splits = walk_forward_splits(100, n_splits=4, expanding=True)
    assert len(splits) == 4
    for train_idx, val_idx in splits:
        # Every training index precedes every validation index (no leakage).
        assert train_idx.max() < val_idx.min()
        # No overlap between train and val.
        assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_expanding_vs_rolling():
    exp = walk_forward_splits(100, n_splits=4, expanding=True)
    roll = walk_forward_splits(100, n_splits=4, expanding=False)
    # Expanding train sets grow; rolling stay bounded.
    exp_sizes = [len(tr) for tr, _ in exp]
    assert exp_sizes == sorted(exp_sizes)
    assert exp_sizes[-1] > len(roll[-1][0])


def test_invalid_n_splits():
    with pytest.raises(ValueError):
        walk_forward_splits(100, n_splits=0)


# --- torch-gated end-to-end validation ---
pytest.importorskip("torch")


def _bundle():
    import pandas as pd

    from trading_ml.models.technical.dataset import build_samples

    r = np.random.default_rng(5)
    frames = []
    day = pd.Timestamp("2024-03-01", tz="UTC")
    price, made = 100.0, 0
    while made < 40:
        if day.weekday() < 5:
            idx = pd.date_range(day + pd.Timedelta(hours=14), periods=78, freq="5min")
            c = price + np.cumsum(r.normal(0, 0.15, 78))
            o = c - r.normal(0, 0.05, 78)
            h = np.maximum(o, c) + np.abs(r.normal(0, 0.08, 78))
            low = np.minimum(o, c) - np.abs(r.normal(0, 0.08, 78))
            frames.append(
                pd.DataFrame(
                    {"open": o, "high": h, "low": low, "close": c, "volume": 1.0}, index=idx
                )
            )
            price = c[-1]
            made += 1
        day += pd.Timedelta(days=1)
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return build_samples(
        df, "S", base_timeframe="5m", context_timeframes=["5m", "15m"], window=16, horizon=6
    )


def test_walk_forward_validate_aggregates():
    from trading_ml.models.technical.walkforward import walk_forward_validate

    cfg = {
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
            "early_stopping_patience": 2,
            "loss": "focal",
            "focal_gamma": 1.5,
        },
        "walk_forward": {"n_splits": 2, "expanding": True},
    }
    result = walk_forward_validate(_bundle(), cfg)
    assert result["aggregate"]["n_folds"] == 2
    assert 0.0 <= result["aggregate"]["balanced_accuracy_mean"] <= 1.0
    assert len(result["folds"]) == 2
