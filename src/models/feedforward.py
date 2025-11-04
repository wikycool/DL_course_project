"""Feedforward neural network implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from src.activations import (
    get_activation,
    get_activation_derivative,
    get_activation_pair,
    softmax,
)
from src.losses import get_loss, l2_regularization_gradients


ArrayLike = np.ndarray


@dataclass
class NetworkConfig:
    input_size: int
    hidden_sizes: Sequence[int]
    output_size: int
    activation: str = "relu"
    output_activation: str = "softmax"
    weight_init: str = "xavier"
    l2_coeff: float = 0.0


class FeedForwardNN:
    """Fully-connected feedforward neural network."""

    def __init__(self, config: NetworkConfig | None = None, **kwargs):
        if config is None:
            config = NetworkConfig(**kwargs)  # type: ignore[arg-type]
        else:
            for key, value in kwargs.items():
                setattr(config, key, value)

        self.config = config
        self.hidden_activation, self.hidden_activation_derivative = get_activation_pair(config.activation)

        if config.output_activation.lower() == "softmax":
            self.output_activation = softmax
            self.output_activation_derivative = None
        else:
            self.output_activation, self.output_activation_derivative = get_activation_pair(config.output_activation)

        self.layer_sizes: List[int] = [config.input_size, *config.hidden_sizes, config.output_size]
        self.num_layers = len(self.layer_sizes) - 1

        self.weights: List[ArrayLike] = []
        self.biases: List[ArrayLike] = []
        self._initialize_parameters(config.weight_init)

        # caches populated during forward pass
        self._activations: List[ArrayLike] = []
        self._pre_activations: List[ArrayLike] = []

    # ------------------------------------------------------------------
    # Parameter initialization
    # ------------------------------------------------------------------
    def _initialize_parameters(self, method: str) -> None:
        method = method.lower()
        rng = np.random.default_rng()

        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if method == "xavier":
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                weight = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
            elif method == "he":
                std = np.sqrt(2.0 / fan_in)
                weight = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
            else:  # default random small values
                weight = rng.normal(0.0, 0.01, size=(fan_in, fan_out)).astype(np.float64)

            bias = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(weight)
            self.biases.append(bias)

    # ------------------------------------------------------------------
    # Forward & backward propagation
    # ------------------------------------------------------------------
    def forward(self, x: ArrayLike) -> ArrayLike:
        activations = [x]
        pre_activations = []

        a = x
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self.hidden_activation(z)
            activations.append(a)

        z_final = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z_final)

        if self.config.output_activation.lower() == "softmax":
            a_final = softmax(z_final)
        else:
            assert self.output_activation is not None
            a_final = self.output_activation(z_final)

        activations.append(a_final)

        self._activations = activations
        self._pre_activations = pre_activations

        return a_final

    def backward(self, y_true: ArrayLike, loss_name: str) -> Tuple[List[ArrayLike], List[ArrayLike]]:
        if not self._activations:
            raise RuntimeError("forward must be called before backward.")

        _, loss_derivative = get_loss(loss_name)

        y_pred = self._activations[-1]
        n_samples = y_true.shape[0]

        if loss_name == "cross_entropy" and self.config.output_activation.lower() == "softmax":
            delta = (y_pred - y_true) / n_samples
        else:
            if self.output_activation_derivative is None:
                raise ValueError("Output activation derivative is undefined for custom configuration.")
            delta = loss_derivative(y_true, y_pred) * self.output_activation_derivative(self._pre_activations[-1])

        weight_grads: List[ArrayLike] = []
        bias_grads: List[ArrayLike] = []

        for layer in reversed(range(self.num_layers)):
            a_prev = self._activations[layer]
            weight_grad = a_prev.T @ delta
            bias_grad = np.sum(delta, axis=0, keepdims=True)

            weight_grads.insert(0, weight_grad)
            bias_grads.insert(0, bias_grad)

            if layer > 0:
                delta = (delta @ self.weights[layer].T) * self.hidden_activation_derivative(self._pre_activations[layer - 1])

        # Normalize gradients by batch size
        weight_grads = [grad / n_samples for grad in weight_grads]
        bias_grads = [grad / n_samples for grad in bias_grads]

        if self.config.l2_coeff > 0:
            l2_grads = l2_regularization_gradients(tuple(self.weights), self.config.l2_coeff)
            weight_grads = [grad + reg for grad, reg in zip(weight_grads, l2_grads)]

        return weight_grads, bias_grads

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        return self.forward(x)

    def predict(self, x: ArrayLike) -> ArrayLike:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------
    def parameters(self) -> List[Tuple[str, ArrayLike]]:
        params = []
        for idx, weight in enumerate(self.weights, start=1):
            params.append((f"W{idx}", weight))
        for idx, bias in enumerate(self.biases, start=1):
            params.append((f"b{idx}", bias))
        return params

    def update_parameters(self, weight_grads: List[ArrayLike], bias_grads: List[ArrayLike], optimizer) -> None:
        for idx, (w, grad) in enumerate(zip(self.weights, weight_grads), start=1):
            updated = optimizer.update_param(w, grad, f"W{idx}")
            self.weights[idx - 1] = updated
        for idx, (b, grad) in enumerate(zip(self.biases, bias_grads), start=1):
            updated = optimizer.update_param(b, grad, f"b{idx}")
            self.biases[idx - 1] = updated

    def flatten_parameters(self) -> ArrayLike:
        return np.concatenate([w.ravel() for w in self.weights])

    def gradient_norm(self, weight_grads: List[ArrayLike]) -> float:
        total = sum(np.sum(grad ** 2) for grad in weight_grads)
        return float(np.sqrt(total))
