"""Backtesting: event-driven engine + performance metrics."""

from trading_ml.backtest.engine import BacktestResult, Trade, run_backtest
from trading_ml.backtest.metrics import compute_metrics

__all__ = ["BacktestResult", "Trade", "run_backtest", "compute_metrics"]
