"""Classification evaluation metrics for Model 1.

Pure-NumPy implementations (no scikit-learn dependency) so evaluation runs in
the lightweight CI environment. Covers the 3-class direction problem
(down / flat / up) with balanced accuracy, per-class precision/recall/F1, and a
confusion matrix.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["down", "flat", "up"]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Rows = true class, cols = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    pred_totals = cm.sum(axis=0).astype(float)
    true_totals = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_totals, out=np.zeros_like(tp), where=pred_totals > 0)
    recall = np.divide(tp, true_totals, out=np.zeros_like(tp), where=true_totals > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3
) -> dict[str, float]:
    """Return a flat dict of accuracy, balanced accuracy, macro-F1 and per-class stats."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1 = _precision_recall_f1(cm)

    n = len(y_true)
    accuracy = float((y_true == y_pred).mean()) if n else 0.0
    # Balanced accuracy = mean per-class recall (robust to class imbalance).
    balanced_accuracy = float(recall.mean())

    report: dict[str, float] = {
        "n": float(n),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1.mean()),
    }
    for i in range(n_classes):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        report[f"precision_{name}"] = float(precision[i])
        report[f"recall_{name}"] = float(recall[i])
        report[f"f1_{name}"] = float(f1[i])
    return report


def format_confusion_matrix(cm: np.ndarray, class_names: list[str] | None = None) -> str:
    """Render a confusion matrix as a small aligned text table."""
    names = class_names or CLASS_NAMES[: cm.shape[0]]
    header = "true\\pred".ljust(10) + "".join(n.rjust(8) for n in names)
    lines = [header]
    for i, name in enumerate(names):
        row = name.ljust(10) + "".join(str(int(v)).rjust(8) for v in cm[i])
        lines.append(row)
    return "\n".join(lines)
