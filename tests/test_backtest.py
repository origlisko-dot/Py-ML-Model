"""Backtest engine + metrics, including no-lookahead execution."""

import numpy as np
import pandas as pd

from trading_ml.backtest import run_backtest
from trading_ml.backtest.metrics import compute_metrics, max_drawdown
from trading_ml.models.base import Signal
from trading_ml.models.risk import RiskModel


def _frame(closes, highs, lows):
    idx = pd.date_range("2024-06-03 14:00", periods=len(closes), freq="5min", tz="UTC")
    o = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": o, "high": highs, "low": lows, "close": closes, "volume": 1.0}, index=idx
    )


def test_entry_next_bar_and_exit_after_entry():
    closes = np.linspace(100, 105, 40)
    df = _frame(closes, closes + 0.3, closes - 0.3)
    # Signal at index 10 (close-labeled timestamp present in the frame).
    sig = Signal(timestamp=df.index[10], symbol="X", direction=1, probability=0.9, horizon=10)
    res = run_backtest([sig], {"X": df}, RiskModel(stop_atr_mult=1.0, rr_ratio=1.5), atr_window=5)
    assert len(res.trades) == 1
    tr = res.trades[0]
    # Entry fills at the NEXT bar (index 11), never the signal bar itself.
    assert tr.entry_time == df.index[11]
    assert tr.exit_time >= tr.entry_time


def test_long_take_profit_positive_pnl():
    # Strongly rising series -> long should reach take-profit for positive PnL.
    closes = np.linspace(100, 130, 60)
    df = _frame(closes, closes + 0.5, closes - 0.1)
    sig = Signal(timestamp=df.index[5], symbol="X", direction=1, probability=0.9, horizon=40)
    res = run_backtest([sig], {"X": df}, RiskModel(stop_atr_mult=1.0, rr_ratio=2.0), atr_window=5)
    assert len(res.trades) == 1
    assert res.trades[0].reason in ("take", "horizon")
    assert res.trades[0].pnl > 0


def test_one_position_per_symbol():
    closes = np.linspace(100, 105, 60)
    df = _frame(closes, closes + 0.3, closes - 0.3)
    # Two signals close in time; the second must be skipped while first is open.
    sigs = [
        Signal(timestamp=df.index[5], symbol="X", direction=1, probability=0.9, horizon=30),
        Signal(timestamp=df.index[6], symbol="X", direction=1, probability=0.9, horizon=30),
    ]
    res = run_backtest(sigs, {"X": df}, RiskModel(), atr_window=5)
    assert len(res.trades) <= 1


def test_metrics_math():
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    equity = np.array([100.0, 102.0, 100.98, 104.01, 101.93, 102.95])
    m = compute_metrics(returns, equity)
    assert m["n_trades"] == 5
    assert 0 <= m["win_rate"] <= 1
    assert m["profit_factor"] > 0


def test_max_drawdown():
    equity = np.array([100.0, 120.0, 90.0, 110.0])
    dd = max_drawdown(equity)
    # Peak 120 -> trough 90 = -25%.
    assert abs(dd - (-0.25)) < 1e-9


def test_trailing_stop_locks_in_gain():
    # Rise to 125 then pull back; a 2*ATR trailing stop should exit in profit.
    closes = np.concatenate([np.linspace(100, 125, 40), np.linspace(125, 110, 25)])
    df = _frame(closes, closes + 0.3, closes - 0.3)
    sig = Signal(timestamp=df.index[10], symbol="X", direction=1, probability=0.9, horizon=60)
    res = run_backtest(
        [sig],
        {"X": df},
        RiskModel(stop_atr_mult=2.0, rr_ratio=20.0),
        atr_window=5,
        trail_atr_mult=2.0,
    )
    assert len(res.trades) == 1
    assert res.trades[0].reason == "trail"
    assert res.trades[0].pnl > 0


def test_portfolio_mode_limits_concurrency():
    from trading_ml.models.risk.portfolio import PortfolioRiskManager

    frames, sigs = {}, []
    for s in ("A", "B", "C"):
        c = np.linspace(100, 110, 60)
        frames[s] = _frame(c, c + 0.3, c - 0.3)
        sigs.append(
            Signal(
                timestamp=frames[s].index[10], symbol=s, direction=1, probability=0.9, horizon=40
            )
        )
    pm = PortfolioRiskManager(
        max_concurrent_positions=2, max_total_risk_pct=1.0, max_symbol_exposure_pct=1.0
    )
    res = run_backtest(
        sigs, frames, RiskModel(stop_atr_mult=1.0, rr_ratio=2.0), atr_window=5, portfolio=pm
    )
    # Only 2 can be open at once; the 3rd simultaneous signal is rejected.
    assert res.rejections.get("max concurrent positions", 0) >= 1
    assert len(res.trades) == 2
