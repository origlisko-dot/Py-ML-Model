"""LightGBM baseline for Model 1.

A gradient-boosting reference on flattened multi-timeframe features (last bar +
window mean per feature per timeframe). Useful as a sanity check and ensemble
candidate for the deep cross-timeframe network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.models.base import BaseModel, Signal
from trading_ml.models.technical.dataset import (
    SampleBundle,
    build_samples,
    concat_bundles,
    temporal_split,
)

_CLASS_TO_DIR = {0: -1, 1: 0, 2: 1}


def flatten_features(bundle: SampleBundle) -> np.ndarray:
    """(N, n_tf, W, F) -> (N, n_tf * F * 2): last-bar value and window mean."""
    if len(bundle) == 0:
        return np.empty((0, 0), dtype=np.float32)
    last = bundle.X[:, :, -1, :]  # (N, n_tf, F)
    mean = bundle.X.mean(axis=2)  # (N, n_tf, F)
    feats = np.concatenate([last, mean], axis=2)  # (N, n_tf, 2F)
    return feats.reshape(len(bundle), -1).astype(np.float32)


class TechnicalBaseline(BaseModel):
    name = "technical-baseline"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: Any = None
        self.feature_names: list[str] | None = None

    def _bundle(self, base_dfs: dict[str, pd.DataFrame]) -> SampleBundle:
        d = self.config["data"]
        lab = self.config["labeling"]
        bundles = [
            build_samples(
                df,
                symbol,
                base_timeframe=d["base_timeframe"],
                context_timeframes=d["context_timeframes"],
                window=d["window"],
                horizon=d["horizon"],
                take_profit_atr=lab["take_profit_atr"],
                stop_loss_atr=lab["stop_loss_atr"],
                atr_window=lab["atr_window"],
                feature_names=self.feature_names,
            )
            for symbol, df in base_dfs.items()
        ]
        return concat_bundles(bundles)

    def fit(self, base_dfs: dict[str, pd.DataFrame]) -> TechnicalBaseline:
        import lightgbm as lgb

        bundle = self._bundle(base_dfs)
        self.feature_names = bundle.feature_names
        train_b, val_b = temporal_split(bundle, self.config["train"]["val_fraction"])
        bcfg = self.config["baseline"]
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=bcfg["n_estimators"],
            learning_rate=bcfg["learning_rate"],
            num_leaves=bcfg["num_leaves"],
            max_depth=bcfg["max_depth"],
            class_weight="balanced",
            verbose=-1,
        )
        eval_set = None
        if len(val_b) > 0:
            eval_set = [(flatten_features(val_b), val_b.y)]
        self.model.fit(flatten_features(train_b), train_b.y, eval_set=eval_set)
        return self

    def probs_for_bundle(self, bundle) -> np.ndarray:
        """Class probabilities (N, 3) for a pre-built sample bundle."""
        if self.model is None:
            raise RuntimeError("Baseline not fitted.")
        return self.model.predict_proba(flatten_features(bundle))

    def predict(self, base_dfs: dict[str, pd.DataFrame], min_prob: float = 0.0) -> list[Signal]:
        if self.model is None:
            raise RuntimeError("Baseline not fitted.")
        signals: list[Signal] = []
        horizon = self.config["data"]["horizon"]
        for symbol, df in base_dfs.items():
            bundle = self._bundle({symbol: df})
            if len(bundle) == 0:
                continue
            probs = self.model.predict_proba(flatten_features(bundle))
            cls = probs.argmax(axis=1)
            for i in range(len(bundle)):
                p = float(probs[i, cls[i]])
                if p < min_prob:
                    continue
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(bundle.times[i]),
                        symbol=symbol,
                        direction=_CLASS_TO_DIR[int(cls[i])],
                        probability=p,
                        horizon=horizon,
                    )
                )
        return signals

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "config": self.config, "feature_names": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TechnicalBaseline:
        import joblib

        blob = joblib.load(path)
        obj = cls(blob["config"])
        obj.model = blob["model"]
        obj.feature_names = blob["feature_names"]
        return obj
