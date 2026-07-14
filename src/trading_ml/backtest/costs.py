"""Trading-cost model for the backtest.

A frictionless backtest overstates performance — especially for high-frequency
day-trading, where commissions and the bid/ask spread compound fast. This model
charges, per trade:

- **commission** — per-share and/or percent-of-notional, with a per-side minimum,
- **slippage** — an adverse price move (bps) modelling market impact / latency,
- **half-spread** — bps of the bid/ask spread paid crossing to trade.

Fills are always *adverse*: a buy fills slightly above the reference price and a
sell slightly below — never in the trader's favour. Slippage and half-spread are
applied once on entry and once on exit; commission is charged on both sides.

The default :class:`CostModel` is all-zeros, so a backtest run without a cost
model behaves exactly as before (frictionless) — full backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

_BPS = 1e-4  # basis points -> fraction


@dataclass
class CostModel:
    commission_per_share: float = 0.0  # $ per share, per side
    commission_pct: float = 0.0  # fraction of notional, per side
    min_commission: float = 0.0  # minimum commission, per side
    slippage_bps: float = 0.0  # adverse price move, bps, per side
    half_spread_bps: float = 0.0  # half bid/ask spread, bps, per side

    @property
    def _per_side_bps(self) -> float:
        return self.slippage_bps + self.half_spread_bps

    def _adverse(self, price: float, is_buy: bool) -> float:
        """Shift ``price`` against the trader: buys up, sells down."""
        move = price * self._per_side_bps * _BPS
        return price + move if is_buy else price - move

    def entry_fill(self, price: float, direction: int) -> float:
        """Fill price when opening: a long buys, a short sells."""
        return self._adverse(price, is_buy=direction > 0)

    def exit_fill(self, price: float, direction: int) -> float:
        """Fill price when closing: a long sells, a short buys."""
        return self._adverse(price, is_buy=direction < 0)

    def commission(self, price: float, quantity: int) -> float:
        """Commission for one side (entry or exit)."""
        variable = self.commission_per_share * quantity + self.commission_pct * price * quantity
        return max(self.min_commission, variable)

    def apply(
        self, entry_price: float, exit_price: float, direction: int, quantity: int
    ) -> tuple[float, float]:
        """Return ``(net_pnl, total_cost)`` for a round-trip trade.

        ``net_pnl`` is PnL at adverse fills, minus entry + exit commissions.
        ``total_cost`` is the frictionless PnL minus ``net_pnl`` — everything the
        costs subtracted (slippage + spread + commissions).
        """
        entry = self.entry_fill(entry_price, direction)
        exit_ = self.exit_fill(exit_price, direction)
        gross = direction * (exit_ - entry) * quantity
        commissions = self.commission(entry, quantity) + self.commission(exit_, quantity)
        net_pnl = gross - commissions

        frictionless = direction * (exit_price - entry_price) * quantity
        return net_pnl, frictionless - net_pnl
