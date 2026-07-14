"""Model 2 — dynamic position sizing.

Given a directional :class:`Signal`, the current price and an ATR volatility
estimate, produces a :class:`TradePlan` with a position size, an ATR-based hard
stop, and a take-profit enforcing a target Risk/Reward ratio.

Three sizing methods are supported (``SizingMethod``):

- **fixed_fractional** — risk a constant fraction of equity per trade.
- **kelly** — size from the signal's win probability and the R/R payoff using a
  (fractional) Kelly criterion; larger edge → larger size.
- **vol_target** — size so the position's ATR-implied risk matches a target
  volatility budget.

A hard notional cap (``max_position_pct``) always applies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from trading_ml.models.base import DIRECTION_FLAT, Signal


class SizingMethod(StrEnum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    VOL_TARGET = "vol_target"


@dataclass
class TradePlan:
    symbol: str
    direction: int
    entry: float
    stop: float
    take_profit: float
    quantity: int
    risk_amount: float
    rr_ratio: float
    timestamp: object = None
    method: str = SizingMethod.FIXED_FRACTIONAL.value
    meta: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.quantity > 0 and self.direction != DIRECTION_FLAT


def kelly_fraction(win_prob: float, rr_ratio: float) -> float:
    """Kelly optimal fraction for a bet paying ``rr_ratio`` on a win.

    ``f* = p - (1 - p) / b`` where ``b`` is the win/loss payoff ratio. Clamped
    to ``[0, 1]`` (never recommend a short-the-edge or all-in bet).
    """
    if rr_ratio <= 0:
        return 0.0
    f = win_prob - (1.0 - win_prob) / rr_ratio
    return max(0.0, min(1.0, f))


class RiskModel:
    def __init__(
        self,
        risk_pct: float = 0.01,
        rr_ratio: float = 2.0,
        stop_atr_mult: float = 1.5,
        max_position_pct: float = 0.25,
        method: str | SizingMethod = SizingMethod.FIXED_FRACTIONAL,
        kelly_multiplier: float = 0.5,
        target_volatility: float = 0.01,
    ):
        if not 0 < risk_pct < 1:
            raise ValueError("risk_pct must be in (0, 1)")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.stop_atr_mult = stop_atr_mult
        self.max_position_pct = max_position_pct
        self.method = SizingMethod(method)
        self.kelly_multiplier = kelly_multiplier  # fractional-Kelly damping
        self.target_volatility = target_volatility

    def _risk_budget(self, signal: Signal, equity: float, atr: float) -> tuple[float, dict]:
        """Dollar risk to allocate to this trade, plus method metadata."""
        if self.method is SizingMethod.KELLY:
            # Fractional Kelly: damp the optimal fraction; the notional cap
            # (max_position_pct) provides the hard safety ceiling.
            f = self.kelly_multiplier * kelly_fraction(signal.probability, self.rr_ratio)
            return equity * f, {"kelly_f": f}
        if self.method is SizingMethod.VOL_TARGET:
            # Risk budget that targets the configured volatility fraction of equity.
            return equity * self.target_volatility, {"target_vol": self.target_volatility}
        return equity * self.risk_pct, {}

    def size_position(
        self,
        signal: Signal,
        equity: float,
        price: float,
        atr: float,
    ) -> TradePlan | None:
        """Return a sized :class:`TradePlan`, or ``None`` for no trade."""
        if signal.direction == DIRECTION_FLAT or atr <= 0 or price <= 0 or equity <= 0:
            return None

        stop_distance = self.stop_atr_mult * atr
        if stop_distance <= 0:
            return None

        if signal.direction > 0:  # long
            stop = price - stop_distance
            take_profit = price + self.rr_ratio * stop_distance
        else:  # short
            stop = price + stop_distance
            take_profit = price - self.rr_ratio * stop_distance

        risk_budget, meta = self._risk_budget(signal, equity, atr)
        quantity = math.floor(risk_budget / stop_distance) if risk_budget > 0 else 0

        # Cap notional exposure regardless of the method-implied size.
        max_qty = math.floor((equity * self.max_position_pct) / price)
        quantity = max(0, min(quantity, max_qty))
        if quantity <= 0:
            return None

        return TradePlan(
            symbol=signal.symbol,
            direction=signal.direction,
            entry=price,
            stop=stop,
            take_profit=take_profit,
            quantity=quantity,
            risk_amount=quantity * stop_distance,
            rr_ratio=self.rr_ratio,
            timestamp=signal.timestamp,
            method=self.method.value,
            meta=meta,
        )
