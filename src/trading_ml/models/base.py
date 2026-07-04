"""Shared model contract.

All three models expose the same surface so the strategy/backtest engines can
treat them uniformly: a model is fit on historical data and emits :class:`Signal`
objects. Model 1 (technical) is the reference full implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Direction encoding shared across the codebase.
DIRECTION_DOWN = -1
DIRECTION_FLAT = 0
DIRECTION_UP = 1


@dataclass
class Signal:
    """A directional trading signal at a point in time.

    ``direction`` ∈ {-1, 0, +1}; ``probability`` is the model's confidence in the
    chosen direction (0..1); ``horizon`` is the intended holding period in bars.
    """

    timestamp: pd.Timestamp
    symbol: str
    direction: int
    probability: float
    horizon: int
    price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction != DIRECTION_FLAT


class BaseModel(ABC):
    """Base class for all trading models."""

    name: str = "base"

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Train the model. Returns ``self``."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> list[Signal]:
        """Emit signals for the given input."""

    def save(self, path: str | Path) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:  # pragma: no cover
        raise NotImplementedError
