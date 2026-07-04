"""Model 3 — event-driven catalyst detection (scaffold)."""

from trading_ml.models.events.catalyst import (
    CatalystClassifier,
    CatalystSignal,
    CatalystType,
    KeywordCatalystClassifier,
)

__all__ = [
    "CatalystClassifier",
    "CatalystSignal",
    "CatalystType",
    "KeywordCatalystClassifier",
]
