"""Trailing stop logic."""

from trading_ml.models.risk.stops import TrailingStop


def test_long_stop_ratchets_up_and_never_down():
    ts = TrailingStop(direction=1, stop=98.0, trail_atr_mult=1.0)
    assert ts.update(high=105, low=104, atr=2.0) == 103.0  # 105 - 1*2
    # Pullback must NOT loosen the stop.
    assert ts.update(high=103, low=101, atr=2.0) == 103.0


def test_short_stop_ratchets_down():
    ts = TrailingStop(direction=-1, stop=102.0, trail_atr_mult=1.0)
    assert ts.update(high=96, low=95, atr=2.0) == 97.0  # 95 + 1*2
    assert ts.update(high=99, low=98, atr=2.0) == 97.0  # no loosening


def test_is_hit_long():
    ts = TrailingStop(direction=1, stop=100.0)
    assert ts.is_hit(high=105, low=99.9)  # low breaches
    assert not ts.is_hit(high=105, low=100.5)


def test_zero_atr_keeps_stop():
    ts = TrailingStop(direction=1, stop=98.0)
    assert ts.update(high=110, low=109, atr=0.0) == 98.0
