"""Optimizers supporting multiple update rules and weight decay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class OptimizerConfig:
    optimizer_type: str = "sgd"
    learning_rate: float = 1e-3
    momentum: float = 0.9
    beta: float = 0.9  # RMSprop decay
    beta1: float = 0.9  # Adam/Nadam first moment
    beta2: float = 0.999  # Adam/Nadam second moment
    epsilon: float = 1e-8
    weight_decay: float = 0.0


class Optimizer:
    """Optimizer supporting SGD, Momentum, Nesterov, RMSprop, Adam, and Nadam."""

    def __init__(self, config: OptimizerConfig | None = None, **overrides):
        cfg = config or OptimizerConfig()
        for key, value in overrides.items():
            setattr(cfg, key, value)

        self.config = cfg
        self._v: Dict[str, ArrayLike] = {}
        self._s: Dict[str, ArrayLike] = {}
        self._t: Dict[str, int] = {}

    def reset_state(self) -> None:
        self._v.clear()
        self._s.clear()
        self._t.clear()

    def _ensure_state(self, name: str, shape: Tuple[int, ...]) -> None:
        if name not in self._v:
            self._v[name] = np.zeros(shape, dtype=np.float64)
        if name not in self._s:
            self._s[name] = np.zeros(shape, dtype=np.float64)
        if name not in self._t:
            self._t[name] = 0

    def update_param(self, param: ArrayLike, grad: ArrayLike, name: str) -> ArrayLike:
        """Apply an optimization step for a single parameter tensor."""
        cfg = self.config
        grad_reg = grad + cfg.weight_decay * param

        self._ensure_state(name, grad.shape)

        opt = cfg.optimizer_type.lower()

        if opt == "momentum":
            self._v[name] = cfg.momentum * self._v[name] - cfg.learning_rate * grad_reg
            return param + self._v[name]

        if opt == "nesterov":
            v_prev = self._v[name].copy()
            self._v[name] = cfg.momentum * self._v[name] - cfg.learning_rate * grad_reg
            return param - cfg.momentum * v_prev + (1 + cfg.momentum) * self._v[name]

        if opt == "rmsprop":
            self._s[name] = cfg.beta * self._s[name] + (1 - cfg.beta) * (grad_reg ** 2)
            return param - cfg.learning_rate * grad_reg / (np.sqrt(self._s[name]) + cfg.epsilon)

        if opt in {"adam", "nadam"}:
            self._t[name] += 1
            self._v[name] = cfg.beta1 * self._v[name] + (1 - cfg.beta1) * grad_reg
            self._s[name] = cfg.beta2 * self._s[name] + (1 - cfg.beta2) * (grad_reg ** 2)
            v_hat = self._v[name] / (1 - cfg.beta1 ** self._t[name])
            s_hat = self._s[name] / (1 - cfg.beta2 ** self._t[name])

            if opt == "adam":
                step = v_hat
            else:  # Nadam
                step = cfg.beta1 * v_hat + (1 - cfg.beta1) * grad_reg

            return param - cfg.learning_rate * step / (np.sqrt(s_hat) + cfg.epsilon)

        if opt == "sgd":
            return param - cfg.learning_rate * grad_reg

        raise ValueError(f"Unsupported optimizer '{cfg.optimizer_type}'.")

    def update_sequence(self, params: Iterable[Tuple[str, ArrayLike, ArrayLike]]):
        """Convenience helper to update a collection of parameters in-place."""
        for name, param, grad in params:
            updated = self.update_param(param, grad, name)
            param[...] = updated


def create_optimizer(**kwargs) -> Optimizer:
    """Factory helper mirroring the reference project API."""
    return Optimizer(**kwargs)
