"""Event-driven backtest engine.

Consumes :class:`Signal` objects (from Model 1), sizes each with the
:class:`RiskModel` (Model 2), and simulates trades bar-by-bar against the
base-timeframe OHLCV. Execution is realistic in the small ways that matter:

- entries fill at the **next bar's open** (never the signal bar's close — that
  would be lookahead),
- exits trigger intrabar on the **stop**/**trailing stop** or **take-profit**
  (stop checked first when both are touched in one bar), else at the horizon
  bar's close,
- **legacy mode** (no portfolio manager): one position per symbol at a time,
  sequential equity — the original behaviour,
- **portfolio mode** (pass a ``PortfolioRiskManager``): many concurrent
  positions, opened/closed in strict time order, gated by portfolio-level
  constraints (concurrency, total risk, exposure, correlation, daily loss).
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_ml.backtest.costs import CostModel
from trading_ml.backtest.metrics import compute_metrics
from trading_ml.features.indicators import atr as atr_indicator
from trading_ml.models.base import Signal
from trading_ml.models.risk.portfolio import PortfolioRiskManager
from trading_ml.models.risk.position_sizing import RiskModel
from trading_ml.models.risk.stops import TrailingStop


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
    reason: str  # "stop" | "trail" | "take" | "horizon"
    cost: float = 0.0  # slippage + spread + commissions charged on this trade


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _simulate_trade(
    frame: pd.DataFrame,
    atr_arr: np.ndarray,
    entry_idx: int,
    plan_dir: int,
    entry_price: float,
    stop: float,
    take: float,
    horizon: int,
    trail_atr_mult: float | None = None,
) -> tuple[float, pd.Timestamp, str]:
    """Walk forward from entry, returning (exit_price, exit_time, reason)."""
    end = min(entry_idx + horizon, len(frame) - 1)
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    closes = frame["close"].to_numpy()
    times = frame.index

    trailing = TrailingStop(plan_dir, stop, trail_atr_mult) if trail_atr_mult else None
    for j in range(entry_idx, end + 1):
        cur_stop = stop
        stop_reason = "stop"
        if trailing is not None:
            cur_stop = trailing.update(highs[j], lows[j], float(atr_arr[j]))
            stop_reason = "trail"
        if plan_dir > 0:
            if lows[j] <= cur_stop:
                return cur_stop, times[j], stop_reason
            if highs[j] >= take:
                return take, times[j], "take"
        else:
            if highs[j] >= cur_stop:
                return cur_stop, times[j], stop_reason
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
    trail_atr_mult: float | None = None,
    portfolio: PortfolioRiskManager | None = None,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Run the backtest over signals and close-labeled base OHLCV frames."""
    risk_model = risk_model or RiskModel()
    cost_model = cost_model or CostModel()  # default: frictionless (backward compatible)
    atr_series = {
        sym: atr_indicator(frame, atr_window).to_numpy() for sym, frame in price_frames.items()
    }
    ordered = sorted(
        (s for s in signals if s.is_actionable and s.probability >= min_prob),
        key=lambda s: s.timestamp,
    )
    if portfolio is None:
        return _run_legacy(
            ordered,
            price_frames,
            atr_series,
            risk_model,
            initial_equity,
            trail_atr_mult,
            periods_per_year,
            cost_model,
        )
    return _run_portfolio(
        ordered,
        price_frames,
        atr_series,
        risk_model,
        portfolio,
        initial_equity,
        trail_atr_mult,
        periods_per_year,
        cost_model,
    )


def _open_context(sig, frame, atr_arr):
    """Return (entry_idx, entry_price, atr_at_signal) or None if not tradable."""
    if sig.timestamp not in frame.index:
        return None
    loc = frame.index.get_loc(sig.timestamp)
    if not isinstance(loc, int):
        return None
    entry_idx = loc + 1
    if entry_idx >= len(frame):
        return None
    atr_val = float(atr_arr[loc])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    return entry_idx, float(frame["open"].iloc[entry_idx]), atr_val


def _run_legacy(
    signals, price_frames, atr_series, risk_model, initial_equity, trail_atr_mult, ppy, cost_model
):
    result = BacktestResult(equity_curve=[initial_equity])
    equity = initial_equity
    busy_until: dict[str, pd.Timestamp] = {}

    for sig in signals:
        frame = price_frames.get(sig.symbol)
        if frame is None:
            continue
        if sig.symbol in busy_until and sig.timestamp < busy_until[sig.symbol]:
            continue
        ctx = _open_context(sig, frame, atr_series[sig.symbol])
        if ctx is None:
            continue
        entry_idx, entry_price, atr_val = ctx
        plan = risk_model.size_position(sig, equity, entry_price, atr_val)
        if plan is None or not plan.is_valid:
            continue
        exit_price, exit_time, reason = _simulate_trade(
            frame,
            atr_series[sig.symbol],
            entry_idx,
            plan.direction,
            entry_price,
            plan.stop,
            plan.take_profit,
            sig.horizon,
            trail_atr_mult,
        )
        pnl, cost = cost_model.apply(entry_price, exit_price, plan.direction, plan.quantity)
        equity += pnl
        result.trades.append(
            Trade(
                sig.symbol,
                plan.direction,
                frame.index[entry_idx],
                exit_time,
                entry_price,
                exit_price,
                plan.quantity,
                pnl,
                pnl / (equity - pnl),
                reason,
                cost,
            )
        )
        result.equity_curve.append(equity)
        busy_until[sig.symbol] = exit_time

    returns = np.array([t.return_pct for t in result.trades], dtype=float)
    result.metrics = compute_metrics(returns, np.array(result.equity_curve), ppy)
    result.metrics["total_costs"] = float(sum(t.cost for t in result.trades))
    return result


def _run_portfolio(
    signals,
    price_frames,
    atr_series,
    risk_model,
    portfolio,
    initial_equity,
    trail_atr_mult,
    ppy,
    cost_model,
):
    result = BacktestResult(equity_curve=[initial_equity])
    equity = initial_equity
    counter = itertools.count()
    # min-heap of (exit_time, seq, Trade) — realized in strict time order.
    pending: list[tuple[pd.Timestamp, int, Trade]] = []

    def flush_until(t):
        nonlocal equity
        while pending and pending[0][0] <= t:
            _, _, tr = heapq.heappop(pending)
            equity_before = equity
            equity += tr.pnl
            tr.return_pct = tr.pnl / equity_before if equity_before else 0.0
            portfolio.register_close(tr.symbol, tr.pnl, tr.exit_time)
            result.trades.append(tr)
            result.equity_curve.append(equity)

    for sig in signals:
        flush_until(sig.timestamp)  # realize any closes due before this signal
        frame = price_frames.get(sig.symbol)
        if frame is None:
            continue
        ctx = _open_context(sig, frame, atr_series[sig.symbol])
        if ctx is None:
            continue
        entry_idx, entry_price, atr_val = ctx
        plan = risk_model.size_position(sig, equity, entry_price, atr_val)
        if plan is None or not plan.is_valid:
            continue
        decision = portfolio.can_open(plan, sig.timestamp, equity)
        if not decision.ok:
            result.rejections[decision.reason] = result.rejections.get(decision.reason, 0) + 1
            continue
        exit_price, exit_time, reason = _simulate_trade(
            frame,
            atr_series[sig.symbol],
            entry_idx,
            plan.direction,
            entry_price,
            plan.stop,
            plan.take_profit,
            sig.horizon,
            trail_atr_mult,
        )
        pnl, cost = cost_model.apply(entry_price, exit_price, plan.direction, plan.quantity)
        portfolio.register_open(plan, sig.timestamp)
        trade = Trade(
            sig.symbol,
            plan.direction,
            frame.index[entry_idx],
            exit_time,
            entry_price,
            exit_price,
            plan.quantity,
            pnl,
            0.0,
            reason,
            cost,
        )
        heapq.heappush(pending, (exit_time, next(counter), trade))

    flush_until(pd.Timestamp.max.tz_localize("UTC"))  # close everything left open

    returns = np.array([t.return_pct for t in result.trades], dtype=float)
    result.metrics = compute_metrics(returns, np.array(result.equity_curve), ppy)
    result.metrics["total_costs"] = float(sum(t.cost for t in result.trades))
    return result
