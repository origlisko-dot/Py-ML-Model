"""Global seeding for reproducible training and tuning."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    ``deterministic_torch`` enables deterministic cuDNN kernels where possible;
    it trades a little speed for run-to-run reproducibility, which matters for
    hyperparameter tuning and walk-forward comparisons.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
