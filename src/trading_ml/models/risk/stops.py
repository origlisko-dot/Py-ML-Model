"""Trailing-stop logic for Model 2.

A trailing stop ratchets in the trade's favor as price moves, locking in gains
while never loosening. For a long, the stop only ever rises; for a short, it
only ever falls. The distance is ATR-scaled so it adapts to volatility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrailingStop:
    """Stateful ATR trailing stop for one open position.

    ``update`` is called once per bar with that bar's high/low/ATR and returns
    the current (possibly tightened) stop level. It never moves against the
    position.
    """

    direction: int  # +1 long, -1 short
    stop: float  # current stop level (initialized to the hard stop)
    trail_atr_mult: float = 2.0

    def update(self, high: float, low: float, atr: float) -> float:
        if atr <= 0:
            return self.stop
        distance = self.trail_atr_mult * atr
        if self.direction > 0:  # long: raise stop under the running high
            candidate = high - distance
            if candidate > self.stop:
                self.stop = candidate
        else:  # short: lower stop over the running low
            candidate = low + distance
            if candidate < self.stop:
                self.stop = candidate
        return self.stop

    def is_hit(self, high: float, low: float) -> bool:
        """Whether this bar's range breaches the current stop."""
        if self.direction > 0:
            return low <= self.stop
        return high >= self.stop
