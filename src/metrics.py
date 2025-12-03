"""Evaluation metrics and reporting utilities."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


ArrayLike = np.ndarray


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute classification accuracy."""
    pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    return float(np.mean(pred_classes == true_classes))


def confusion_matrix_counts(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """Return the raw confusion matrix counts."""
    pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    return confusion_matrix(true_classes, pred_classes)


def classification_report_dict(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Return a classification report as a dictionary."""
    pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    return classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True)


def classification_report_text(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: List[str] | None = None,
) -> str:
    """Return a human-readable classification report."""
    pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    return classification_report(true_classes, pred_classes, target_names=class_names)
