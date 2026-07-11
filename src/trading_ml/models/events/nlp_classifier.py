"""Model 3 — transformer catalyst classifier.

Drops in behind :class:`~trading_ml.models.events.catalyst.CatalystClassifier`
using two Hugging Face pipelines:

* **Zero-shot** (default ``facebook/bart-large-mnli``) picks the catalyst
  *type* by scoring each headline against natural-language hypotheses — one per
  :class:`CatalystType`. No labeled catalyst dataset is needed.
* **FinBERT sentiment** (``ProsusAI/finbert``) sets the *strength* of the
  expected move: a strongly-worded story (positive or negative) scales the
  per-type base strength up; a muted or neutral one scales it down. The
  polarity is stored in ``meta["sentiment"]`` for future directional use.

Both pipelines are imported and loaded lazily (``uv sync --extra nlp``), so the
module imports cleanly under the lightweight CI where ``transformers`` is absent.
"""

from __future__ import annotations

from typing import Any

from trading_ml.data.providers.base import NewsItem
from trading_ml.models.events.catalyst import (
    CatalystClassifier,
    CatalystSignal,
    CatalystType,
)

# Defaults mirror config/models/catalyst.yaml so the classifier is usable
# without a config dict (e.g. `TransformerCatalystClassifier()`).
_DEFAULT_TYPE_MODEL = "facebook/bart-large-mnli"
_DEFAULT_SENTIMENT_MODEL = "ProsusAI/finbert"
_DEFAULT_MIN_CONFIDENCE = 0.5
_DEFAULT_LABELS: dict[str, str] = {
    "merger or acquisition or takeover": "m_and_a",
    "strategic partnership or collaboration or joint venture": "partnership",
    "technological breakthrough, product launch, or regulatory approval": "breakthrough",
    "quarterly earnings or financial results or guidance": "earnings",
    "regulatory action, lawsuit, or government investigation": "regulatory",
    "routine business news with no major catalyst": "none",
}
_DEFAULT_STRENGTH: dict[str, float] = {
    "m_and_a": 0.9,
    "breakthrough": 0.7,
    "partnership": 0.5,
    "earnings": 0.4,
    "regulatory": 0.5,
}


class TransformerCatalystClassifier(CatalystClassifier):
    """Zero-shot catalyst typing + FinBERT strength scoring."""

    name = "nlp-catalyst"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        mcfg = config.get("model", {})
        self.type_model: str = mcfg.get("type_model", _DEFAULT_TYPE_MODEL)
        self.sentiment_model: str = mcfg.get("sentiment_model", _DEFAULT_SENTIMENT_MODEL)
        self.min_confidence: float = mcfg.get("min_confidence", _DEFAULT_MIN_CONFIDENCE)
        self.multi_label: bool = mcfg.get("multi_label", False)

        label_map = config.get("catalyst_labels", _DEFAULT_LABELS)
        # hypothesis string -> CatalystType
        self._label_to_type: dict[str, CatalystType] = {
            hyp: CatalystType(ctype) for hyp, ctype in label_map.items()
        }
        self._hypotheses: list[str] = list(self._label_to_type)
        self._strength: dict[str, float] = config.get("strength", _DEFAULT_STRENGTH)

        # Lazily constructed HF pipelines (see _ensure_loaded).
        self._zero_shot: Any = None
        self._sentiment: Any = None

    # -- pipeline loading ---------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._zero_shot is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - exercised only without nlp extra
            raise ImportError(
                "transformers is required for the NLP catalyst classifier. "
                "Install with `uv sync --extra nlp`."
            ) from exc
        self._zero_shot = pipeline("zero-shot-classification", model=self.type_model)
        self._sentiment = pipeline("sentiment-analysis", model=self.sentiment_model)

    # -- classification ------------------------------------------------------ #
    def classify(self, item: NewsItem) -> CatalystSignal:
        return self.classify_batch([item])[0]

    def classify_batch(self, items: list[NewsItem]) -> list[CatalystSignal]:
        if not items:
            return []
        self._ensure_loaded()
        texts = [_text_of(it) for it in items]

        zs = self._zero_shot(texts, self._hypotheses, multi_label=self.multi_label)
        # transformers returns a single dict for one input, a list otherwise.
        if isinstance(zs, dict):
            zs = [zs]

        sentiments = self._sentiment(texts)
        if isinstance(sentiments, dict):
            sentiments = [sentiments]

        return [
            self._build_signal(item, z, sent)
            for item, z, sent in zip(items, zs, sentiments, strict=True)
        ]

    def _build_signal(self, item: NewsItem, zs_result: dict, sentiment: dict) -> CatalystSignal:
        top_label = zs_result["labels"][0]
        confidence = float(zs_result["scores"][0])
        ctype = self._label_to_type.get(top_label, CatalystType.NONE)

        if ctype is CatalystType.NONE or confidence < self.min_confidence:
            return CatalystSignal(
                timestamp=item.timestamp,
                symbol=item.symbol,
                catalyst=CatalystType.NONE,
                strength=0.0,
                confidence=0.0,
                source=item.source,
            )

        base = self._strength.get(ctype.value, 0.3)
        strength = round(min(1.0, base * _sentiment_factor(sentiment)), 4)
        return CatalystSignal(
            timestamp=item.timestamp,
            symbol=item.symbol,
            catalyst=ctype,
            strength=strength,
            confidence=round(confidence, 4),
            source=item.source,
            meta={
                "sentiment": str(sentiment.get("label", "")).lower(),
                "sentiment_score": round(float(sentiment.get("score", 0.0)), 4),
                "classifier": self.name,
            },
        )


def _text_of(item: NewsItem) -> str:
    """Combine headline and body into one string for the models."""
    body = item.body.strip()
    return f"{item.headline}. {body}" if body else item.headline


def _sentiment_factor(sentiment: dict) -> float:
    """Scale strength by sentiment conviction.

    ``strength`` is the expected *magnitude* of the move, not its direction, so
    a strongly negative story is as much a catalyst as a strongly positive one.
    Neutral sentiment dampens the base strength.
    """
    label = str(sentiment.get("label", "")).lower()
    score = float(sentiment.get("score", 0.0))
    if label == "neutral":
        return 0.5
    # positive / negative: 0.5 at low conviction, ~1.0 at high conviction.
    return 0.5 + 0.5 * score
