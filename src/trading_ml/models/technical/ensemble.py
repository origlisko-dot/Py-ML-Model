"""Ensemble of the cross-timeframe network and the LightGBM baseline.

The deep network and the gradient-boosting baseline make different kinds of
errors; averaging their class probabilities usually yields a steadier signal
than either alone. Both share the exact same sample construction (feature set
and windows), so their probabilities align sample-for-sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.models.base import BaseModel, Signal
from trading_ml.models.technical.baseline import TechnicalBaseline
from trading_ml.models.technical.module import TechnicalModel

_CLASS_TO_DIR = {0: -1, 1: 0, 2: 1}


class EnsembleTechnicalModel(BaseModel):
    """Probability-averaging ensemble: ``w * net + (1 - w) * baseline``."""

    name = "technical-ensemble"

    def __init__(self, config: dict[str, Any], net_weight: float = 0.5):
        if not 0.0 <= net_weight <= 1.0:
            raise ValueError("net_weight must be in [0, 1]")
        self.config = config
        self.net_weight = net_weight
        self.net_model = TechnicalModel(config)
        self.baseline = TechnicalBaseline(config)

    def fit(self, base_dfs: dict[str, pd.DataFrame], logger: Any = None) -> EnsembleTechnicalModel:
        self.net_model.fit(base_dfs, logger=logger)
        # Share the network's feature set so both models build identical bundles.
        self.baseline.feature_names = self.net_model.feature_names
        self.baseline.fit(base_dfs)
        return self

    def _ensemble_probs(self, bundle) -> np.ndarray:
        p_net = self.net_model.probs_for_bundle(bundle)
        p_base = self.baseline.probs_for_bundle(bundle)
        return self.net_weight * p_net + (1.0 - self.net_weight) * p_base

    def predict(self, base_dfs: dict[str, pd.DataFrame], min_prob: float = 0.0) -> list[Signal]:
        horizon = self.config["data"]["horizon"]
        signals: list[Signal] = []
        for symbol, df in base_dfs.items():
            bundle = self.net_model._bundle_from_dfs({symbol: df})
            if len(bundle) == 0:
                continue
            probs = self._ensemble_probs(bundle)
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
                        meta={"probs": probs[i].tolist(), "ensemble": True},
                    )
                )
        return signals

    def save(self, path: str | Path) -> None:
        """Save both members under a directory (``<path>/net.pt`` + baseline)."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self.net_model.save(d / "net.pt")
        self.baseline.save(d / "baseline.joblib")
        import json

        (d / "ensemble.json").write_text(json.dumps({"net_weight": self.net_weight}))

    @classmethod
    def load(cls, path: str | Path) -> EnsembleTechnicalModel:
        import json

        d = Path(path)
        net = TechnicalModel.load(d / "net.pt")
        obj = cls(
            net.config, net_weight=json.loads((d / "ensemble.json").read_text())["net_weight"]
        )
        obj.net_model = net
        obj.baseline = TechnicalBaseline.load(d / "baseline.joblib")
        return obj
