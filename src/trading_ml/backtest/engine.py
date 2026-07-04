"""Event-driven backtest engine.

Consumes :class:`Signal` objects (from Model 1), sizes each with the
:class:`RiskModel` (Model 2), and simulates the trade bar-by-bar against the
base-timeframe OHLCV. Execution is realistic in the small ways that matter:

- entries fill at the **next bar's open** (never the signal bar's close — that
  would be lookahead),
- exits trigger intrabar on the **stop** or **take-profit** (stop checked first
  when both are touched in one bar), else at the horizon bar's close,
- one position per symbol at a time (no pyramiding), sequential equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_ml.backtest.metrics import compute_metrics
from trading_ml.features.indicators import atr as atr_indicator
from trading_ml.models.base import Signal
from trading_ml.models.risk.position_sizing import RiskModel


@dataclass
class Trade:
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    quantity: int
    pnl: float
    return_pct: float
    reason: str  # "stop" | "take" | "horizon"


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _simulate_trade(
    frame: pd.DataFrame,
    entry_idx: int,
    plan_dir: int,
    entry_price: float,
    stop: float,
    take: float,
    horizon: int,
) -> tuple[float, pd.Timestamp, str]:
    """Walk forward from entry, returning (exit_price, exit_time, reason)."""
    end = min(entry_idx + horizon, len(frame) - 1)
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    closes = frame["close"].to_numpy()
    times = frame.index

    for j in range(entry_idx, end + 1):
        if plan_dir > 0:
            if lows[j] <= stop:
                return stop, times[j], "stop"
            if highs[j] >= take:
                return take, times[j], "take"
        else:
            if highs[j] >= stop:
                return stop, times[j], "stop"
            if lows[j] <= take:
                return take, times[j], "take"
    return closes[end], times[end], "horizon"


def run_backtest(
    signals: list[Signal],
    price_frames: dict[str, pd.DataFrame],
    risk_model: RiskModel | None = None,
    initial_equity: float = 100_000.0,
    atr_window: int = 14,
    min_prob: float = 0.0,
    periods_per_year: float = 252.0,
) -> BacktestResult:
    """Run the backtest over signals and close-labeled base OHLCV frames."""
    risk_model = risk_model or RiskModel()
    result = BacktestResult(equity_curve=[initial_equity])
    equity = initial_equity

    # Precompute ATR and integer positions per symbol.
    atr_series: dict[str, pd.Series] = {}
    for sym, frame in price_frames.items():
        atr_series[sym] = atr_indicator(frame, atr_window)

    # Order signals by time; enforce one open position per symbol.
    signals = sorted(
        (s for s in signals if s.is_actionable and s.probability >= min_prob),
        key=lambda s: s.timestamp,
    )
    busy_until: dict[str, pd.Timestamp] = {}

    for sig in signals:
        frame = price_frames.get(sig.symbol)
        if frame is None or sig.timestamp not in frame.index:
            continue
        if sig.symbol in busy_until and sig.timestamp < busy_until[sig.symbol]:
            continue  # position still open

        loc = frame.index.get_loc(sig.timestamp)
        if not isinstance(loc, int):
            continue
        entry_idx = loc + 1  # fill at next bar open (no lookahead)
        if entry_idx >= len(frame):
            continue

        entry_price = float(frame["open"].iloc[entry_idx])
        atr_val = float(atr_series[sig.symbol].iloc[loc])
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        plan = risk_model.size_position(sig, equity, entry_price, atr_val)
        if plan is None or not plan.is_valid:
            continue

        exit_price, exit_time, reason = _simulate_trade(
            frame, entry_idx, plan.direction, entry_price, plan.stop, plan.take_profit, sig.horizon
        )
        pnl = plan.direction * (exit_price - entry_price) * plan.quantity
        ret = pnl / equity
        equity += pnl

        result.trades.append(
            Trade(
                symbol=sig.symbol,
                direction=plan.direction,
                entry_time=frame.index[entry_idx],
                exit_time=exit_time,
                entry=entry_price,
                exit=exit_price,
                quantity=plan.quantity,
                pnl=pnl,
                return_pct=ret,
                reason=reason,
            )
        )
        result.equity_curve.append(equity)
        busy_until[sig.symbol] = exit_time

    returns = np.array([t.return_pct for t in result.trades], dtype=float)
    result.metrics = compute_metrics(returns, np.array(result.equity_curve), periods_per_year)
    return result
