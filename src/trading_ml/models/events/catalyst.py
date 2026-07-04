"""Model 3 — event-driven catalyst detection (scaffold).

Scans business/news text to flag *catalysts* — M&A, strategic partnerships,
technological breakthroughs — that may move a stock before the market fully
prices them in. This module defines the stable interfaces and a lightweight
keyword baseline so the rest of the system can integrate the event signal today;
a transformer/LLM classifier drops in behind :class:`CatalystClassifier` later
(``uv sync --extra nlp``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from trading_ml.data.providers.base import NewsItem


class CatalystType(StrEnum):
    M_AND_A = "m_and_a"
    PARTNERSHIP = "partnership"
    BREAKTHROUGH = "breakthrough"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    NONE = "none"


@dataclass
class CatalystSignal:
    """A detected catalyst tied to a point in time."""

    timestamp: pd.Timestamp
    symbol: str
    catalyst: CatalystType
    strength: float  # 0..1 expected magnitude of the move
    confidence: float  # 0..1 classifier confidence
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.catalyst is not CatalystType.NONE and self.confidence > 0.0


class CatalystClassifier(ABC):
    """Classifies a news item into a catalyst type with strength/confidence."""

    name: str = "base-catalyst"

    @abstractmethod
    def classify(self, item: NewsItem) -> CatalystSignal:
        """Classify a single news item."""

    def classify_batch(self, items: list[NewsItem]) -> list[CatalystSignal]:
        return [self.classify(i) for i in items]


# Keyword lexicon for the baseline classifier. A real model replaces this.
_KEYWORDS: dict[CatalystType, tuple[str, ...]] = {
    CatalystType.M_AND_A: ("acquire", "acquisition", "merger", "takeover", "buyout", "to buy"),
    CatalystType.PARTNERSHIP: ("partnership", "collaborat", "strategic alliance", "joint venture"),
    CatalystType.BREAKTHROUGH: ("breakthrough", "fda approval", "patent", "unveils", "milestone"),
    CatalystType.EARNINGS: ("earnings", "beats estimates", "guidance", "revenue"),
    CatalystType.REGULATORY: ("sec", "regulator", "antitrust", "investigation", "lawsuit"),
}


class KeywordCatalystClassifier(CatalystClassifier):
    """Deterministic keyword baseline — a placeholder for the NLP model.

    Not intended for production; it exists so the strategy layer and tests can
    exercise the event pathway before the transformer classifier is trained.
    """

    name = "keyword-catalyst"

    def classify(self, item: NewsItem) -> CatalystSignal:
        text = f"{item.headline} {item.body}".lower()
        best_type = CatalystType.NONE
        best_hits = 0
        for ctype, words in _KEYWORDS.items():
            hits = sum(1 for w in words if w in text)
            if hits > best_hits:
                best_hits, best_type = hits, ctype

        if best_hits == 0:
            return CatalystSignal(
                timestamp=item.timestamp,
                symbol=item.symbol,
                catalyst=CatalystType.NONE,
                strength=0.0,
                confidence=0.0,
                source=item.source,
            )
        confidence = min(1.0, 0.4 + 0.2 * best_hits)
        strength = {
            CatalystType.M_AND_A: 0.9,
            CatalystType.BREAKTHROUGH: 0.7,
            CatalystType.PARTNERSHIP: 0.5,
            CatalystType.EARNINGS: 0.4,
            CatalystType.REGULATORY: 0.5,
        }.get(best_type, 0.3)
        return CatalystSignal(
            timestamp=item.timestamp,
            symbol=item.symbol,
            catalyst=best_type,
            strength=strength,
            confidence=confidence,
            source=item.source,
        )
