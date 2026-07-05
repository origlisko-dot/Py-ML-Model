"""Classification evaluation metrics (pure NumPy — runs without torch)."""

import numpy as np

from trading_ml.models.technical.evaluation import (
    classification_report,
    confusion_matrix,
    format_confusion_matrix,
)


def test_confusion_matrix_shape_and_counts():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == 6
    assert cm[0, 0] == 2  # both class-0 correct
    assert cm[1, 2] == 1  # one class-1 predicted as class-2


def test_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    rep = classification_report(y, y)
    assert rep["accuracy"] == 1.0
    assert rep["balanced_accuracy"] == 1.0
    assert rep["macro_f1"] == 1.0


def test_balanced_accuracy_handles_imbalance():
    # 90 class-0, 10 class-1; predict all class-0.
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    rep = classification_report(y_true, y_pred, n_classes=2)
    assert rep["accuracy"] == 0.9
    # balanced accuracy = mean recall = (1.0 + 0.0)/2 = 0.5, unfooled by imbalance
    assert abs(rep["balanced_accuracy"] - 0.5) < 1e-9


def test_report_has_per_class_keys():
    y = np.array([0, 1, 2])
    rep = classification_report(y, y)
    for name in ("down", "flat", "up"):
        assert f"precision_{name}" in rep
        assert f"recall_{name}" in rep
        assert f"f1_{name}" in rep


def test_format_confusion_matrix_renders():
    cm = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]))
    text = format_confusion_matrix(cm)
    assert "down" in text and "up" in text
