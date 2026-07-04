"""Performance metrics for backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading periods per year for Sharpe annualization (≈252 trading days).
_ANNUAL_TRADING_DAYS = 252


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (fraction, ≤ 0)."""
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def compute_metrics(
    trade_returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: float = _ANNUAL_TRADING_DAYS,
) -> dict[str, float]:
    """Summary metrics from per-trade returns and the resulting equity curve."""
    trade_returns = np.asarray(trade_returns, dtype=float)
    equity_curve = np.asarray(equity_curve, dtype=float)

    n = trade_returns.size
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    total_return = float(equity_curve[-1] / equity_curve[0] - 1.0) if equity_curve.size else 0.0

    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0

    return {
        "n_trades": float(n),
        "total_return": total_return,
        "win_rate": float(wins.size / n) if n else 0.0,
        "avg_return": float(trade_returns.mean()) if n else 0.0,
        "profit_factor": profit_factor,
        "sharpe": _sharpe(trade_returns, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "best_trade": float(trade_returns.max()) if n else 0.0,
        "worst_trade": float(trade_returns.min()) if n else 0.0,
    }


def metrics_to_frame(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
