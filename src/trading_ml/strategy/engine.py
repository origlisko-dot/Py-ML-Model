"""Strategy engine — fuse the three models into trade decisions.

Pipeline: Model 1 emits directional :class:`Signal` objects → the strategy
optionally boosts/filters them with Model 3 catalyst signals → Model 2 sizes each
survivor in the backtest engine. Model 2/3 are scaffolds today, so catalyst
fusion is off unless catalysts are supplied.
"""

from __future__ import annotations

import pandas as pd

from trading_ml.models.base import Signal
from trading_ml.models.events.catalyst import CatalystSignal


class StrategyEngine:
    def __init__(
        self,
        min_prob: float = 0.4,
        catalyst_window: str = "1D",
        catalyst_boost: float = 0.15,
    ):
        self.min_prob = min_prob
        self.catalyst_window = pd.Timedelta(catalyst_window)
        self.catalyst_boost = catalyst_boost

    def combine(
        self,
        model_signals: list[Signal],
        catalysts: list[CatalystSignal] | None = None,
    ) -> list[Signal]:
        """Return actionable signals after catalyst fusion and probability gating."""
        catalysts = catalysts or []
        by_symbol: dict[str, list[CatalystSignal]] = {}
        for c in catalysts:
            if c.is_actionable:
                by_symbol.setdefault(c.symbol, []).append(c)

        out: list[Signal] = []
        for sig in model_signals:
            if not sig.is_actionable:
                continue
            prob = sig.probability
            boost = self._catalyst_boost(sig, by_symbol.get(sig.symbol, []))
            if boost > 0:
                prob = min(1.0, prob + boost)
                sig = Signal(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    direction=sig.direction,
                    probability=prob,
                    horizon=sig.horizon,
                    price=sig.price,
                    meta={**sig.meta, "catalyst_boost": boost},
                )
            if prob >= self.min_prob:
                out.append(sig)
        return out

    def _catalyst_boost(self, sig: Signal, catalysts: list[CatalystSignal]) -> float:
        """Boost when a recent catalyst precedes the signal (same symbol)."""
        best = 0.0
        for c in catalysts:
            delta = sig.timestamp - c.timestamp
            if pd.Timedelta(0) <= delta <= self.catalyst_window:
                best = max(best, self.catalyst_boost * c.strength * c.confidence)
        return best
