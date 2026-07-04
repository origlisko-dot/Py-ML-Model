"""Model 2 — dynamic risk management (scaffold + working baseline).

Capital-preservation first. Given a directional :class:`Signal`, the current
price and an ATR volatility estimate, it produces a :class:`TradePlan` with:

- a **fixed-fractional** position size (risk at most ``risk_pct`` of equity),
- an **ATR-based hard stop**, and
- a **take-profit** enforcing a target **Risk/Reward** ratio.

This is the baseline implementation behind a clean interface; advanced sizing
(Kelly, volatility targeting, correlation-aware exposure) plugs in here later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_ml.models.base import DIRECTION_FLAT, Signal


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

    @property
    def is_valid(self) -> bool:
        return self.quantity > 0 and self.direction != DIRECTION_FLAT


class RiskModel:
    def __init__(
        self,
        risk_pct: float = 0.01,
        rr_ratio: float = 2.0,
        stop_atr_mult: float = 1.5,
        max_position_pct: float = 0.25,
    ):
        if not 0 < risk_pct < 1:
            raise ValueError("risk_pct must be in (0, 1)")
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.stop_atr_mult = stop_atr_mult
        self.max_position_pct = max_position_pct

    def size_position(
        self,
        signal: Signal,
        equity: float,
        price: float,
        atr: float,
    ) -> TradePlan | None:
        """Return a sized :class:`TradePlan`, or ``None`` for no trade.

        ``None`` is returned for flat signals, non-positive ATR/price, or when
        the risk budget rounds the position down to zero shares.
        """
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

        risk_budget = equity * self.risk_pct
        quantity = math.floor(risk_budget / stop_distance)

        # Cap notional exposure regardless of the stop-implied size.
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
        )
