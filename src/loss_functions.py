"""Loss functions and regularization utilities."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


ArrayLike = np.ndarray
LossFn = Callable[[ArrayLike, ArrayLike], float]
LossDerivativeFn = Callable[[ArrayLike, ArrayLike], ArrayLike]


def mse_loss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error loss."""
    return float(np.mean((y_true - y_pred) ** 2))


def mse_derivative(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """Derivative of MSE with respect to predictions."""
    n_samples = y_true.shape[0]
    return -2.0 * (y_true - y_pred) / n_samples


def cross_entropy_loss(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """Cross entropy loss for one-hot labels and probability predictions."""
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


def cross_entropy_derivative(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> ArrayLike:
    """Derivative of cross entropy with respect to predictions."""
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    n_samples = y_true.shape[0]
    return -(y_true / y_pred) / n_samples


def l2_regularization(weights: Tuple[ArrayLike, ...], l2_coeff: float) -> float:
    """Compute the L2 regularization penalty."""
    if l2_coeff <= 0:
        return 0.0
    return float(l2_coeff * sum(np.sum(w ** 2) for w in weights))


def l2_regularization_gradients(weights: Tuple[ArrayLike, ...], l2_coeff: float) -> Tuple[ArrayLike, ...]:
    """Derivative of the L2 penalty for each weight tensor."""
    if l2_coeff <= 0:
        return tuple(np.zeros_like(w) for w in weights)
    return tuple(2.0 * l2_coeff * w for w in weights)


_LOSSES: Dict[str, Tuple[LossFn, LossDerivativeFn]] = {
    "mse": (mse_loss, mse_derivative),
    "mean_squared_error": (mse_loss, mse_derivative),
    "cross_entropy": (cross_entropy_loss, cross_entropy_derivative),
}


def get_loss(name: str) -> Tuple[LossFn, LossDerivativeFn]:
    """Return loss function and derivative pair by name."""
    key = name.lower()
    if key not in _LOSSES:
        raise ValueError(f"Unsupported loss '{name}'. Available: {list(_LOSSES)}")
    return _LOSSES[key]


def compute_loss(y_pred: ArrayLike, y_true: ArrayLike, name: str) -> float:
    """Compatibility wrapper used by legacy scripts."""
    loss_fn, _ = get_loss(name)
    return loss_fn(y_true, y_pred)
