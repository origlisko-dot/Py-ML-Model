"""Transformer catalyst classifier — mapping / strength / gating logic.

Real HF pipelines (bart-large-mnli, FinBERT) are multi-GB downloads that need
network, so they are unsuitable for unit tests. Instead we inject fake pipelines
into the classifier instance (bypassing the lazy `_ensure_loaded`) and assert on
the deterministic logic: label→type mapping, min-confidence gating, and how
FinBERT sentiment scales strength. This runs under the lightweight CI too, since
`nlp_classifier` imports `transformers` lazily.
"""

import pandas as pd

from trading_ml.data.providers.base import NewsItem
from trading_ml.models.events import CatalystType, TransformerCatalystClassifier


def _news(headline: str, symbol: str = "AAPL") -> NewsItem:
    return NewsItem(
        timestamp=pd.Timestamp("2024-06-03 12:00", tz="UTC"), symbol=symbol, headline=headline
    )


class _FakeZeroShot:
    """Returns a fixed (label, score) as the top prediction for every text."""

    def __init__(self, label: str, score: float):
        self.label, self.score = label, score

    def __call__(self, texts, hypotheses, multi_label=False):
        one = {"labels": [self.label, *hypotheses], "scores": [self.score, 0.0]}
        return [one for _ in texts] if isinstance(texts, list) else one


class _FakeSentiment:
    def __init__(self, label: str, score: float):
        self.label, self.score = label, score

    def __call__(self, texts):
        one = {"label": self.label, "score": self.score}
        return [one for _ in texts] if isinstance(texts, list) else one


def _clf(zs_label, zs_score, sent_label="positive", sent_score=0.9):
    clf = TransformerCatalystClassifier()
    clf._zero_shot = _FakeZeroShot(zs_label, zs_score)
    clf._sentiment = _FakeSentiment(sent_label, sent_score)
    return clf


# The default hypothesis strings mapped to each type (from catalyst.yaml defaults).
_MNA = "merger or acquisition or takeover"
_NONE = "routine business news with no major catalyst"


def test_maps_top_label_to_type():
    sig = _clf(_MNA, 0.92).classify(_news("Acme to acquire Beta"))
    assert sig.catalyst is CatalystType.M_AND_A
    assert sig.is_actionable
    assert sig.confidence == 0.92
    assert sig.meta["sentiment"] == "positive"


def test_below_min_confidence_is_none():
    sig = _clf(_MNA, 0.30).classify(_news("maybe a deal?"))  # default min_confidence 0.5
    assert sig.catalyst is CatalystType.NONE
    assert not sig.is_actionable


def test_neutral_label_is_none():
    sig = _clf(_NONE, 0.99).classify(_news("company holds annual picnic"))
    assert sig.catalyst is CatalystType.NONE


def test_sentiment_scales_strength():
    strong = _clf(_MNA, 0.9, "positive", 0.99).classify(_news("huge takeover")).strength
    muted = _clf(_MNA, 0.9, "neutral", 0.99).classify(_news("takeover talks")).strength
    assert strong > muted  # conviction scales strength; neutral dampens it
    assert 0.0 < strong <= 1.0


def test_batch_alignment():
    clf = _clf(_MNA, 0.8)
    out = clf.classify_batch([_news("a"), _news("b"), _news("c")])
    assert len(out) == 3
    assert all(s.catalyst is CatalystType.M_AND_A for s in out)


def test_empty_batch():
    assert _clf(_MNA, 0.8).classify_batch([]) == []
