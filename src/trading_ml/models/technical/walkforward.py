"""Walk-forward (rolling-origin) cross-validation for Model 1.

A single train/val split can flatter a time-series model. Walk-forward
validation instead trains on the past and validates on the *next* chronological
block, repeatedly rolling the origin forward. It reports out-of-sample metrics
aggregated across folds — a far more honest estimate of live performance.

Each fold trains a fresh network with a compact Torch loop (early-stopped on the
validation balanced accuracy) so the whole sweep stays fast and self-contained.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trading_ml.models.technical.dataset import SampleBundle, apply_scaler, fit_scaler
from trading_ml.models.technical.evaluation import classification_report
from trading_ml.models.technical.module import _build_loss, _class_weights
from trading_ml.models.technical.network import CrossTimeframeNet


def walk_forward_splits(
    n_samples: int, n_splits: int, expanding: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, val_idx)`` pairs over ``n_samples`` chronological rows.

    The series is cut into ``n_splits + 1`` equal blocks. Fold ``i`` validates on
    block ``i + 1`` using blocks ``0..i`` (expanding) or block ``i`` (rolling) for
    training. Indices are contiguous and time-ordered — never shuffled.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    bounds = np.linspace(0, n_samples, n_splits + 2, dtype=int)
    splits = []
    for i in range(1, n_splits + 1):
        val_lo, val_hi = bounds[i], bounds[i + 1]
        train_lo = 0 if expanding else bounds[i - 1]
        train_idx = np.arange(train_lo, val_lo)
        val_idx = np.arange(val_lo, val_hi)
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def _train_one_fold(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    n_features: int,
    n_timeframes: int,
    config: dict[str, Any],
    seed: int = 42,
) -> dict[str, float]:
    torch.manual_seed(seed)
    ncfg = config["network"]
    tcfg = config["train"]
    net = CrossTimeframeNet(
        n_features=n_features,
        n_timeframes=n_timeframes,
        n_classes=3,
        encoder=ncfg["encoder"],
        d_model=ncfg["d_model"],
        n_heads=ncfg["n_heads"],
        encoder_layers=ncfg["encoder_layers"],
        dropout=ncfg["dropout"],
    )
    loss_fn = _build_loss(tcfg.get("loss", "ce"), tcfg.get("focal_gamma", 1.5), _class_weights(ytr))
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=tcfg["batch_size"],
        shuffle=True,
    )
    xva_t = torch.from_numpy(Xva)

    best_bal, best_report, patience = -1.0, {}, tcfg.get("early_stopping_patience", 4)
    stale = 0
    for _ in range(tcfg["max_epochs"]):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(xva_t).argmax(dim=1).numpy()
        report = classification_report(yva, pred)
        if report["balanced_accuracy"] > best_bal:
            best_bal, best_report, stale = report["balanced_accuracy"], report, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_report


def walk_forward_validate(
    bundle: SampleBundle,
    config: dict[str, Any],
    n_splits: int | None = None,
    expanding: bool | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run walk-forward CV on a time-sorted bundle; return per-fold + aggregate metrics."""
    wf = config.get("walk_forward", {})
    n_splits = n_splits if n_splits is not None else wf.get("n_splits", 4)
    expanding = expanding if expanding is not None else wf.get("expanding", True)

    splits = walk_forward_splits(len(bundle), n_splits, expanding)
    fold_reports: list[dict[str, float]] = []
    for tr_idx, va_idx in splits:
        tr = _subset(bundle, tr_idx)
        va = _subset(bundle, va_idx)
        mean, std = fit_scaler(tr)  # fit scaler on the fold's TRAIN only
        Xtr = apply_scaler(tr, mean, std)
        Xva = apply_scaler(va, mean, std)
        report = _train_one_fold(
            Xtr, tr.y, Xva, va.y, bundle.n_features, len(bundle.timeframes), config, seed
        )
        fold_reports.append(report)

    return {"folds": fold_reports, "aggregate": _aggregate(fold_reports)}


def _subset(bundle: SampleBundle, idx: np.ndarray) -> SampleBundle:
    return SampleBundle(
        bundle.X[idx],
        bundle.y[idx],
        bundle.times[idx],
        bundle.symbols[idx],
        bundle.feature_names,
        bundle.timeframes,
    )


def _aggregate(reports: list[dict[str, float]]) -> dict[str, float]:
    if not reports:
        return {}
    keys = ["balanced_accuracy", "macro_f1", "accuracy"]
    agg: dict[str, float] = {"n_folds": float(len(reports))}
    for k in keys:
        vals = np.array([r.get(k, 0.0) for r in reports], dtype=float)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std())
    return agg


# Silence unused-import complaints for nn (kept for type clarity in signatures).
_ = nn
