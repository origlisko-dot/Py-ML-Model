"""Reproducible seeding."""

import numpy as np

from trading_ml.seeding import set_seed


def test_numpy_determinism():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b)
