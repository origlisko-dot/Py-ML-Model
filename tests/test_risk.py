"""Model 2 — position sizing / risk management."""

import pandas as pd

from trading_ml.models.base import Signal
from trading_ml.models.risk import RiskModel


def _signal(direction: int) -> Signal:
    return Signal(
        timestamp=pd.Timestamp("2024-06-03 14:00", tz="UTC"),
        symbol="AAPL",
        direction=direction,
        probability=0.7,
        horizon=12,
    )


def test_long_plan_geometry():
    rm = RiskModel(risk_pct=0.01, rr_ratio=2.0, stop_atr_mult=1.5)
    plan = rm.size_position(_signal(1), equity=100_000, price=100.0, atr=1.0)
    assert plan is not None and plan.is_valid
    assert plan.stop < plan.entry < plan.take_profit
    # R/R geometry: reward distance == rr * risk distance.
    risk = plan.entry - plan.stop
    reward = plan.take_profit - plan.entry
    assert abs(reward / risk - 2.0) < 1e-9


def test_short_plan_geometry():
    rm = RiskModel(risk_pct=0.01, rr_ratio=2.0, stop_atr_mult=1.5)
    plan = rm.size_position(_signal(-1), equity=100_000, price=100.0, atr=1.0)
    assert plan is not None
    assert plan.take_profit < plan.entry < plan.stop


def test_risk_budget_respected():
    rm = RiskModel(risk_pct=0.01, rr_ratio=2.0, stop_atr_mult=1.0)
    equity = 100_000
    plan = rm.size_position(_signal(1), equity=equity, price=50.0, atr=2.0)
    # risk_amount = qty * stop_distance must not exceed 1% of equity.
    assert plan.risk_amount <= equity * 0.01 + 1e-6


def test_flat_signal_returns_none():
    rm = RiskModel()
    assert rm.size_position(_signal(0), 100_000, 100.0, 1.0) is None


def test_zero_atr_returns_none():
    rm = RiskModel()
    assert rm.size_position(_signal(1), 100_000, 100.0, 0.0) is None


def test_max_position_cap():
    rm = RiskModel(risk_pct=0.5, rr_ratio=2.0, stop_atr_mult=0.01, max_position_pct=0.25)
    plan = rm.size_position(_signal(1), equity=100_000, price=100.0, atr=1.0)
    # Notional capped at 25% of equity despite the huge risk budget.
    assert plan.quantity * plan.entry <= 100_000 * 0.25 + 100.0
