"""Ensemble of net + LightGBM baseline (torch/lightgbm-gated)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightgbm")


def _days(n=40, start="2024-03-01", seed=3):
    r = np.random.default_rng(seed)
    frames = []
    day = pd.Timestamp(start, tz="UTC")
    price, made = 100.0, 0
    while made < n:
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
    return df


_CFG = {
    "data": {
        "base_timeframe": "5m",
        "context_timeframes": ["5m", "15m"],
        "window": 16,
        "horizon": 6,
    },
    "labeling": {"take_profit_atr": 1.5, "stop_loss_atr": 1.0, "atr_window": 14},
    "network": {"encoder": "tcn", "d_model": 16, "n_heads": 2, "encoder_layers": 1, "dropout": 0.1},
    "train": {
        "batch_size": 64,
        "max_epochs": 1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "val_fraction": 0.2,
        "early_stopping_patience": 2,
        "num_workers": 0,
        "loss": "focal",
        "focal_gamma": 1.5,
    },
    "baseline": {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15, "max_depth": -1},
}


@pytest.fixture(scope="module")
def frames():
    return {"A": _days(seed=1), "B": _days(seed=2, start="2024-03-02")}


def test_ensemble_fit_predict(frames):
    from trading_ml.models.technical.ensemble import EnsembleTechnicalModel

    ens = EnsembleTechnicalModel(_CFG, net_weight=0.5).fit(frames)
    sigs = ens.predict(frames, min_prob=0.0)
    assert len(sigs) > 0
    assert all(s.direction in (-1, 0, 1) for s in sigs)
    assert all(0.0 <= s.probability <= 1.0 for s in sigs)


def test_ensemble_probs_are_average(frames):
    from trading_ml.models.technical.ensemble import EnsembleTechnicalModel

    ens = EnsembleTechnicalModel(_CFG, net_weight=0.5).fit(frames)
    bundle = ens.net_model._bundle_from_dfs({"A": frames["A"]})
    p_net = ens.net_model.probs_for_bundle(bundle)
    p_base = ens.baseline.probs_for_bundle(bundle)
    p_ens = ens._ensemble_probs(bundle)
    assert np.allclose(p_ens, 0.5 * p_net + 0.5 * p_base, atol=1e-5)
    # Probabilities are valid distributions.
    assert np.allclose(p_ens.sum(axis=1), 1.0, atol=1e-4)


def test_ensemble_save_load(frames, tmp_path):
    from trading_ml.models.technical.ensemble import EnsembleTechnicalModel

    ens = EnsembleTechnicalModel(_CFG, net_weight=0.6).fit(frames)
    ens.save(tmp_path / "ens")
    loaded = EnsembleTechnicalModel.load(tmp_path / "ens")
    assert loaded.net_weight == 0.6
    a = ens.predict(frames, min_prob=0.0)
    b = loaded.predict(frames, min_prob=0.0)
    assert len(a) == len(b)


def test_invalid_net_weight():
    from trading_ml.models.technical.ensemble import EnsembleTechnicalModel

    with pytest.raises(ValueError):
        EnsembleTechnicalModel(_CFG, net_weight=1.5)
