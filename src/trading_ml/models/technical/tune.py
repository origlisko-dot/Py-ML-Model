"""Optuna hyperparameter search for Model 1.

The objective is the **walk-forward** mean balanced accuracy — optimizing the
honest out-of-sample metric rather than a single split. Each trial rebuilds the
sample bundle (barrier geometry is itself tunable) and runs a short walk-forward
sweep. Optional dependency: ``uv sync --extra tune``.
"""

from __future__ import annotations

import copy
from typing import Any

from trading_ml.models.technical.module import TechnicalModel
from trading_ml.models.technical.walkforward import walk_forward_validate


def _suggest_config(trial, base_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["network"]["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
    cfg["network"]["encoder_layers"] = trial.suggest_int("encoder_layers", 1, 3)
    cfg["network"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
    cfg["train"]["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg["train"]["focal_gamma"] = trial.suggest_float("focal_gamma", 0.0, 3.0)
    cfg["labeling"]["take_profit_atr"] = trial.suggest_float("take_profit_atr", 1.0, 3.0)
    cfg["labeling"]["stop_loss_atr"] = trial.suggest_float("stop_loss_atr", 0.5, 2.0)
    return cfg


def tune_hyperparams(
    base_dfs: dict,
    base_config: dict[str, Any],
    n_trials: int = 20,
    timeout_seconds: int = 0,
    tuning_epochs: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """Run an Optuna study; return the best params and score.

    ``tuning_epochs`` caps per-fold epochs so the search stays fast.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optuna is required. Install with `uv sync --extra tune`.") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        cfg = _suggest_config(trial, base_config)
        cfg["train"] = {**cfg["train"], "max_epochs": tuning_epochs}
        # Build the (possibly re-labeled) bundle for this trial.
        scratch = TechnicalModel(cfg)
        bundle = scratch._bundle_from_dfs(base_dfs)
        if len(bundle) < 50:
            return 0.0
        result = walk_forward_validate(bundle, cfg, seed=seed)
        return result["aggregate"].get("balanced_accuracy_mean", 0.0)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds or None,
        show_progress_bar=False,
    )
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
