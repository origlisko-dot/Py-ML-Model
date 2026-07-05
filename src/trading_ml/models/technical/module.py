"""Training/inference wrapper for Model 1.

``LitTechnical`` is the LightningModule (loss, metrics, optimizer).
``TechnicalModel`` is the high-level :class:`BaseModel` used by the CLI and
backtest: it builds samples, does a walk-forward split, fits a train-only
feature scaler, trains the network, and emits :class:`Signal` objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import lightning as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from trading_ml.models.base import BaseModel, Signal
from trading_ml.models.technical.dataset import (
    SampleBundle,
    apply_scaler,
    build_samples,
    concat_bundles,
    fit_scaler,
    temporal_split,
)
from trading_ml.models.technical.network import CrossTimeframeNet

# class index {0,1,2} -> direction {-1,0,+1}
_CLASS_TO_DIR = {0: -1, 1: 0, 2: 1}


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al.) — down-weights easy, well-classified
    samples so training focuses on the hard, ambiguous bars that dominate noisy
    financial data. ``gamma=0`` reduces to weighted cross-entropy.
    """

    def __init__(self, gamma: float = 1.5, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        weight = cast("torch.Tensor | None", self.weight)
        ce = torch.nn.functional.nll_loss(log_prob, target, weight=weight, reduction="none")
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _build_loss(name: str, gamma: float, class_weights) -> nn.Module:
    weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    return nn.CrossEntropyLoss(weight=weight)


class LitTechnical(pl.LightningModule):
    def __init__(
        self,
        net: CrossTimeframeNet,
        lr: float,
        weight_decay: float,
        class_weights=None,
        loss: str = "ce",
        focal_gamma: float = 1.5,
    ):
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = _build_loss(loss, focal_gamma, class_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def _class_weights(y: np.ndarray, n_classes: int = 3) -> list[float]:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    inv = counts.sum() / (n_classes * counts)
    return inv.tolist()


class TechnicalModel(BaseModel):
    name = "technical"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.net: CrossTimeframeNet | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.feature_names: list[str] | None = None
        self.timeframes: list[str] | None = None

    # -- sample building -------------------------------------------------- #
    def _bundle_from_dfs(self, base_dfs: dict[str, pd.DataFrame]) -> SampleBundle:
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

    # -- training --------------------------------------------------------- #
    def fit(self, base_dfs: dict[str, pd.DataFrame], logger: Any = None) -> TechnicalModel:
        bundle = self._bundle_from_dfs(base_dfs)
        self.feature_names = bundle.feature_names
        self.timeframes = bundle.timeframes

        tcfg = self.config["train"]
        train_b, val_b = temporal_split(bundle, tcfg["val_fraction"])
        self.scaler_mean, self.scaler_std = fit_scaler(train_b)

        train_ds = _to_tensor_ds(
            apply_scaler(train_b, self.scaler_mean, self.scaler_std), train_b.y
        )
        val_ds = _to_tensor_ds(apply_scaler(val_b, self.scaler_mean, self.scaler_std), val_b.y)

        ncfg = self.config["network"]
        self.net = CrossTimeframeNet(
            n_features=bundle.n_features,
            n_timeframes=len(bundle.timeframes),
            n_classes=3,
            encoder=ncfg["encoder"],
            d_model=ncfg["d_model"],
            n_heads=ncfg["n_heads"],
            encoder_layers=ncfg["encoder_layers"],
            dropout=ncfg["dropout"],
        )
        lit = LitTechnical(
            self.net,
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
            class_weights=_class_weights(train_b.y),
            loss=tcfg.get("loss", "ce"),
            focal_gamma=tcfg.get("focal_gamma", 1.5),
        )

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg["num_workers"]
        )
        val_loader = (
            DataLoader(val_ds, batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"])
            if len(val_b) > 0
            else None
        )

        callbacks: list[Any] = []
        if val_loader is not None:
            from lightning.pytorch.callbacks import EarlyStopping

            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=tcfg["early_stopping_patience"],
                    mode="min",
                )
            )

        trainer = pl.Trainer(
            max_epochs=tcfg["max_epochs"],
            callbacks=callbacks,
            logger=logger if logger is not None else False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="auto",
            devices=1,
            log_every_n_steps=1,
        )
        trainer.fit(lit, train_loader, val_loader)
        self.net.eval()
        return self

    # -- inference -------------------------------------------------------- #
    @torch.no_grad()
    def probs_for_bundle(self, bundle) -> np.ndarray:
        """Class probabilities (N, 3) for a pre-built sample bundle."""
        if self.net is None or self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Model is not fitted/loaded.")
        self.net.eval()
        X = apply_scaler(bundle, self.scaler_mean, self.scaler_std)
        return torch.softmax(self.net(torch.from_numpy(X)), dim=1).numpy()

    @torch.no_grad()
    def predict(self, base_dfs: dict[str, pd.DataFrame], min_prob: float = 0.0) -> list[Signal]:
        if self.net is None or self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Model is not fitted/loaded.")
        self.net.eval()
        signals: list[Signal] = []
        d = self.config["data"]
        for symbol, df in base_dfs.items():
            bundle = self._bundle_from_dfs({symbol: df})
            if len(bundle) == 0:
                continue
            X = apply_scaler(bundle, self.scaler_mean, self.scaler_std)
            logits = self.net(torch.from_numpy(X))
            probs = torch.softmax(logits, dim=1).numpy()
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
                        horizon=d["horizon"],
                        meta={"probs": probs[i].tolist()},
                    )
                )
        return signals

    # -- persistence ------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": None if self.net is None else self.net.state_dict(),
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "feature_names": self.feature_names,
                "timeframes": self.timeframes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TechnicalModel:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["config"])
        model.feature_names = ckpt["feature_names"]
        model.timeframes = ckpt["timeframes"]
        model.scaler_mean = ckpt["scaler_mean"]
        model.scaler_std = ckpt["scaler_std"]
        ncfg = model.config["network"]
        model.net = CrossTimeframeNet(
            n_features=len(model.feature_names),
            n_timeframes=len(model.timeframes),
            n_classes=3,
            encoder=ncfg["encoder"],
            d_model=ncfg["d_model"],
            n_heads=ncfg["n_heads"],
            encoder_layers=ncfg["encoder_layers"],
            dropout=ncfg["dropout"],
        )
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.eval()
        return model


def _to_tensor_ds(X: np.ndarray, y: np.ndarray):
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
