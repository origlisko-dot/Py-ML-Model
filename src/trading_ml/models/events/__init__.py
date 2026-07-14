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
    "TransformerCatalystClassifier",
]


def __getattr__(name: str):
    """Lazily expose the transformer classifier without importing it eagerly.

    Keeps ``import trading_ml.models.events`` cheap and CI-safe (transformers
    is only pulled in when the NLP classifier is actually referenced).
    """
    if name == "TransformerCatalystClassifier":
        from trading_ml.models.events.nlp_classifier import TransformerCatalystClassifier

        return TransformerCatalystClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
