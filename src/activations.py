"""Activation functions and derivatives used in feedforward networks."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


ArrayLike = np.ndarray
ActivationFn = Callable[[ArrayLike], ArrayLike]
DerivativeFn = Callable[[ArrayLike], ArrayLike]


def relu(x: ArrayLike) -> ArrayLike:
    """Rectified linear unit activation."""
    return np.maximum(0, x)


def relu_derivative(x: ArrayLike) -> ArrayLike:
    """Derivative of the ReLU activation."""
    return (x > 0).astype(x.dtype if isinstance(x, np.ndarray) else float)


def sigmoid(x: ArrayLike) -> ArrayLike:
    """Sigmoid activation function with clipping for numerical stability."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: ArrayLike) -> ArrayLike:
    """Derivative of the sigmoid activation."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: ArrayLike) -> ArrayLike:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def tanh_derivative(x: ArrayLike) -> ArrayLike:
    """Derivative of the hyperbolic tangent."""
    return 1.0 - np.tanh(x) ** 2


def identity(x: ArrayLike) -> ArrayLike:
    """Identity/linear activation."""
    return x


def identity_derivative(x: ArrayLike) -> ArrayLike:
    """Derivative of the identity activation."""
    return np.ones_like(x)


def softmax(x: ArrayLike, axis: int = 1) -> ArrayLike:
    """Softmax activation for the output layer."""
    # Shift by max for numerical stability
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


_ACTIVATIONS: Dict[str, Tuple[ActivationFn, DerivativeFn]] = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "identity": (identity, identity_derivative),
    "linear": (identity, identity_derivative),
}


def get_activation(name: str) -> ActivationFn:
    """Return the activation function for a given name."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Available: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[key][0]


def get_activation_derivative(name: str) -> DerivativeFn:
    """Return the derivative function for a given activation name."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Available: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[key][1]


def get_activation_pair(name: str) -> Tuple[ActivationFn, DerivativeFn]:
    """Return both the activation and its derivative."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Available: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]
