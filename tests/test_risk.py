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


def test_kelly_fraction_formula():
    from trading_ml.models.risk.position_sizing import kelly_fraction

    # p=0.6, b=2 -> 0.6 - 0.4/2 = 0.4
    assert abs(kelly_fraction(0.6, 2.0) - 0.4) < 1e-9
    # negative edge clamps to 0
    assert kelly_fraction(0.3, 1.0) == 0.0
    # bounded to 1
    assert kelly_fraction(1.0, 2.0) <= 1.0


def test_kelly_sizing_scales_with_confidence():
    from trading_ml.models.risk.position_sizing import RiskModel, SizingMethod

    rm = RiskModel(
        method=SizingMethod.KELLY,
        rr_ratio=2.0,
        stop_atr_mult=1.0,
        max_position_pct=1.0,
        kelly_multiplier=0.1,
    )
    # atr=10 keeps sizes below the notional cap so Kelly's edge shows through.
    lo = rm.size_position(_signal(1), 100_000, 100.0, 10.0)
    hi_sig = _signal(1)
    hi_sig.probability = 0.9
    hi = rm.size_position(hi_sig, 100_000, 100.0, 10.0)
    # Higher win probability -> larger Kelly size.
    assert hi.quantity > lo.quantity
    assert hi.method == "kelly" and "kelly_f" in hi.meta


def test_vol_target_sizing():
    from trading_ml.models.risk.position_sizing import RiskModel, SizingMethod

    rm = RiskModel(
        method=SizingMethod.VOL_TARGET,
        stop_atr_mult=1.0,
        max_position_pct=1.0,
        target_volatility=0.01,
    )
    plan = rm.size_position(_signal(1), 100_000, 100.0, 2.0)
    # risk budget = equity * target_vol = 1000; qty = 1000 / (stop_dist=2) = 500
    assert plan.quantity == 500
    assert plan.method == "vol_target"
