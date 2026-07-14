"""Portfolio-level risk manager."""

import numpy as np
import pandas as pd

from trading_ml.models.risk.portfolio import PortfolioRiskManager, correlation_lookup
from trading_ml.models.risk.position_sizing import TradePlan

_T = pd.Timestamp("2024-06-03 14:00", tz="UTC")


def _plan(symbol, quantity=100, risk=300.0, entry=100.0):
    return TradePlan(symbol, 1, entry, 98.0, 104.0, quantity, risk, 2.0, timestamp=_T)


def test_max_concurrent_positions():
    pm = PortfolioRiskManager(
        max_concurrent_positions=1, max_total_risk_pct=1.0, max_symbol_exposure_pct=1.0
    )
    assert pm.can_open(_plan("A"), _T, 100_000).ok
    pm.register_open(_plan("A"), _T)
    rej = pm.can_open(_plan("B"), _T, 100_000)
    assert not rej.ok and "concurrent" in rej.reason


def test_total_risk_budget():
    pm = PortfolioRiskManager(
        max_total_risk_pct=0.01, max_symbol_exposure_pct=1.0, max_concurrent_positions=10
    )
    pm.register_open(_plan("A", risk=300), _T)
    # 300 + 800 = 1100 > 1% of 100k = 1000 -> reject
    rej = pm.can_open(_plan("B", risk=800), _T, 100_000)
    assert not rej.ok and "risk budget" in rej.reason


def test_symbol_exposure_cap():
    pm = PortfolioRiskManager(max_symbol_exposure_pct=0.10, max_total_risk_pct=1.0)
    # notional 300*100 = 30k > 10% of 100k
    rej = pm.can_open(_plan("A", quantity=300), _T, 100_000)
    assert not rej.ok and "exposure" in rej.reason


def test_daily_loss_guardrail():
    pm = PortfolioRiskManager(
        max_daily_loss_pct=0.02, max_total_risk_pct=1.0, max_symbol_exposure_pct=1.0
    )
    pm.register_open(_plan("A"), _T)
    pm.register_close("A", -2500.0, _T)  # -2.5% > 2% limit
    rej = pm.can_open(_plan("B"), _T, 100_000)
    assert not rej.ok and "daily loss" in rej.reason


def test_daily_loss_resets_next_day():
    pm = PortfolioRiskManager(
        max_daily_loss_pct=0.02, max_total_risk_pct=1.0, max_symbol_exposure_pct=1.0
    )
    pm.register_close("A", -2500.0, _T)
    next_day = _T + pd.Timedelta(days=1)
    assert pm.can_open(_plan("B"), next_day, 100_000).ok


def test_correlation_rejection():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    returns = pd.DataFrame(
        {"A": a, "C": a * 0.97 + rng.normal(0, 0.05, 200), "D": rng.normal(0, 1, 200)}
    )
    pm = PortfolioRiskManager(
        max_total_risk_pct=1.0,
        max_symbol_exposure_pct=1.0,
        correlation=correlation_lookup(returns),
        max_correlation=0.8,
    )
    pm.register_open(_plan("A"), _T)
    assert not pm.can_open(_plan("C"), _T, 100_000).ok  # highly correlated
    assert pm.can_open(_plan("D"), _T, 100_000).ok  # uncorrelated
