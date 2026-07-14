"""Model 3 — keyword catalyst classifier scaffold."""

import pandas as pd

from trading_ml.data.providers.base import NewsItem
from trading_ml.models.events import CatalystType, KeywordCatalystClassifier


def _news(headline: str) -> NewsItem:
    return NewsItem(
        timestamp=pd.Timestamp("2024-06-03 12:00", tz="UTC"),
        symbol="AAPL",
        headline=headline,
    )


def test_detects_m_and_a():
    clf = KeywordCatalystClassifier()
    sig = clf.classify(_news("Acme Corp to acquire Beta Inc in $2B takeover"))
    assert sig.catalyst is CatalystType.M_AND_A
    assert sig.is_actionable
    assert 0 < sig.confidence <= 1


def test_detects_breakthrough():
    clf = KeywordCatalystClassifier()
    sig = clf.classify(_news("Company unveils breakthrough after FDA approval"))
    assert sig.catalyst is CatalystType.BREAKTHROUGH


def test_no_catalyst_for_neutral_text():
    clf = KeywordCatalystClassifier()
    sig = clf.classify(_news("The weather was mild in the region today"))
    assert sig.catalyst is CatalystType.NONE
    assert not sig.is_actionable
    assert sig.confidence == 0.0


def test_batch():
    clf = KeywordCatalystClassifier()
    out = clf.classify_batch([_news("merger announced"), _news("nothing here")])
    assert len(out) == 2
    assert out[0].catalyst is CatalystType.M_AND_A


def test_catalyst_boosts_matching_signal():
    """A recent catalyst raises the probability of a same-symbol signal in-window."""
    from trading_ml.models.base import DIRECTION_UP, Signal
    from trading_ml.strategy.engine import StrategyEngine

    catalyst = KeywordCatalystClassifier().classify(_news("Acme to acquire Beta in a takeover"))
    signal = Signal(
        timestamp=pd.Timestamp("2024-06-03 15:00", tz="UTC"),  # 3h after the catalyst
        symbol="AAPL",
        direction=DIRECTION_UP,
        probability=0.5,
        horizon=5,
    )
    engine = StrategyEngine(min_prob=0.4, catalyst_window="1D", catalyst_boost=0.15)
    out = engine.combine([signal], [catalyst])
    assert len(out) == 1
    assert out[0].probability > 0.5  # boosted
    assert "catalyst_boost" in out[0].meta


def test_catalyst_outside_window_does_not_boost():
    from trading_ml.models.base import DIRECTION_UP, Signal
    from trading_ml.strategy.engine import StrategyEngine

    catalyst = KeywordCatalystClassifier().classify(_news("Acme to acquire Beta in a takeover"))
    signal = Signal(
        timestamp=pd.Timestamp("2024-06-10 12:00", tz="UTC"),  # a week later
        symbol="AAPL",
        direction=DIRECTION_UP,
        probability=0.5,
        horizon=5,
    )
    engine = StrategyEngine(min_prob=0.4, catalyst_window="1D", catalyst_boost=0.15)
    out = engine.combine([signal], [catalyst])
    assert out[0].probability == 0.5  # no boost outside the window
