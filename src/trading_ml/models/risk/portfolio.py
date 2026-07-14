"""Portfolio-level risk management for Model 2.

Position sizing decides *how big* a single trade is; the portfolio manager
decides *whether it may open at all* given everything already on. It enforces,
in order:

- **max concurrent positions** — cap the number of simultaneous open trades,
- **total risk budget** — summed open-trade risk must stay under a fraction of
  equity,
- **per-symbol exposure** — cap notional in any single symbol,
- **correlation-aware exposure** — reject a new trade too correlated with an
  existing open position (avoids stacking the same bet),
- **daily-loss guardrail** — halt new entries once the day's realized loss
  breaches a limit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from trading_ml.models.risk.position_sizing import TradePlan


@dataclass
class OpenPosition:
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    risk_amount: float
    notional: float


# A correlation lookup: (symbol_a, symbol_b) -> correlation in [-1, 1].
CorrelationFn = Callable[[str, str], float]


@dataclass
class RejectReason:
    ok: bool
    reason: str = ""


@dataclass
class PortfolioRiskManager:
    max_concurrent_positions: int = 5
    max_total_risk_pct: float = 0.06
    max_symbol_exposure_pct: float = 0.25
    max_daily_loss_pct: float = 0.03
    max_correlation: float = 0.8
    correlation: CorrelationFn | None = None

    _open: dict[str, OpenPosition] = field(default_factory=dict)
    _day: object = None
    _daily_pnl: float = 0.0

    # -- queries ---------------------------------------------------------- #
    @property
    def open_positions(self) -> dict[str, OpenPosition]:
        return dict(self._open)

    def total_risk(self) -> float:
        return sum(p.risk_amount for p in self._open.values())

    def _roll_day(self, timestamp: pd.Timestamp) -> None:
        day = pd.Timestamp(timestamp).normalize()
        if self._day != day:
            self._day = day
            self._daily_pnl = 0.0

    def can_open(self, plan: TradePlan, timestamp: pd.Timestamp, equity: float) -> RejectReason:
        """Decide whether ``plan`` may open under all portfolio constraints."""
        self._roll_day(timestamp)

        if plan.symbol in self._open:
            return RejectReason(False, "symbol already open")
        if len(self._open) >= self.max_concurrent_positions:
            return RejectReason(False, "max concurrent positions")
        if self._daily_pnl <= -abs(self.max_daily_loss_pct) * equity:
            return RejectReason(False, "daily loss limit reached")

        notional = plan.quantity * plan.entry
        if notional > self.max_symbol_exposure_pct * equity + 1e-6:
            return RejectReason(False, "symbol exposure cap")

        if (self.total_risk() + plan.risk_amount) > self.max_total_risk_pct * equity + 1e-6:
            return RejectReason(False, "total risk budget")

        if self.correlation is not None:
            for sym in self._open:
                if abs(self.correlation(plan.symbol, sym)) >= self.max_correlation:
                    return RejectReason(False, f"correlated with open {sym}")

        return RejectReason(True)

    # -- mutations -------------------------------------------------------- #
    def register_open(self, plan: TradePlan, timestamp: pd.Timestamp) -> None:
        self._roll_day(timestamp)
        self._open[plan.symbol] = OpenPosition(
            symbol=plan.symbol,
            direction=plan.direction,
            entry_time=pd.Timestamp(timestamp),
            risk_amount=plan.risk_amount,
            notional=plan.quantity * plan.entry,
        )

    def register_close(self, symbol: str, pnl: float, timestamp: pd.Timestamp) -> None:
        self._roll_day(timestamp)
        self._open.pop(symbol, None)
        self._daily_pnl += pnl


def correlation_lookup(returns: pd.DataFrame) -> CorrelationFn:
    """Build a symbol-pair correlation function from a returns DataFrame.

    ``returns`` columns are symbols. Missing pairs default to 0 (uncorrelated).
    """
    corr = returns.corr()

    def lookup(a: str, b: str) -> float:
        if a == b:
            return 1.0
        try:
            val = corr.loc[a, b]
        except KeyError:
            return 0.0
        return 0.0 if pd.isna(val) else float(val)

    return lookup
