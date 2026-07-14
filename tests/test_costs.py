"""Trading-cost model: adverse fills, commissions, and backtest integration."""

import numpy as np
import pandas as pd

from trading_ml.backtest import run_backtest
from trading_ml.backtest.costs import CostModel
from trading_ml.models.base import Signal
from trading_ml.models.risk import RiskModel


def test_zero_cost_is_frictionless():
    cm = CostModel()  # all zeros
    net, cost = cm.apply(entry_price=100.0, exit_price=110.0, direction=1, quantity=10)
    assert net == 100.0  # (110 - 100) * 10
    assert cost == 0.0


def test_long_costs_reduce_pnl():
    cm = CostModel(commission_per_share=0.01, slippage_bps=10.0, half_spread_bps=0.0)
    net, cost = cm.apply(entry_price=100.0, exit_price=110.0, direction=1, quantity=10)
    # entry fills at 100.1 (buy up), exit at 109.89 (sell down); gross 97.9,
    # commission 0.1 per side -> net 97.7, cost = 100 - 97.7 = 2.3.
    assert abs(net - 97.7) < 1e-9
    assert abs(cost - 2.3) < 1e-9


def test_short_fills_are_adverse_the_other_way():
    cm = CostModel(commission_per_share=0.01, slippage_bps=10.0)
    net, cost = cm.apply(entry_price=100.0, exit_price=90.0, direction=-1, quantity=10)
    # short sells entry at 99.9, buys back exit at 90.09; gross 98.1, comm 0.2 -> 97.9.
    assert abs(net - 97.9) < 1e-9
    assert abs(cost - 2.1) < 1e-9


def test_min_commission_enforced():
    cm = CostModel(commission_per_share=0.001, min_commission=1.0)
    # variable = 0.001 * 10 = 0.01 < min 1.0 -> per side commission is 1.0.
    assert cm.commission(price=100.0, quantity=10) == 1.0


def test_commission_pct_of_notional():
    cm = CostModel(commission_pct=0.001)
    assert abs(cm.commission(price=200.0, quantity=5) - 1.0) < 1e-9  # 0.001 * 200 * 5


def _frame(closes):
    idx = pd.date_range("2024-06-03 14:00", periods=len(closes), freq="5min", tz="UTC")
    o = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": o, "high": o + 0.5, "low": o - 0.1, "close": o, "volume": 1.0}, index=idx
    )


def test_backtest_costs_lower_return():
    closes = np.linspace(100, 130, 60)
    df = _frame(closes)
    sig = Signal(timestamp=df.index[5], symbol="X", direction=1, probability=0.9, horizon=40)
    risk = RiskModel(stop_atr_mult=1.0, rr_ratio=2.0)

    free = run_backtest([sig], {"X": df}, risk, atr_window=5)
    costed = run_backtest(
        [sig],
        {"X": df},
        risk,
        atr_window=5,
        cost_model=CostModel(commission_per_share=0.01, slippage_bps=5.0, half_spread_bps=5.0),
    )

    assert len(free.trades) == len(costed.trades) == 1
    assert free.metrics["total_costs"] == 0.0
    assert costed.metrics["total_costs"] > 0.0
    assert costed.metrics["total_return"] < free.metrics["total_return"]
    assert costed.trades[0].cost > 0.0
