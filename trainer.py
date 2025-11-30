"""Training utilities for feedforward networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from data_loader import create_batches, one_hot_encode
from metrics import accuracy
from loss_functions import get_loss, l2_regularization

try:  # Optional dependency for logging
    import wandb  # type: ignore
except Exception:  # pragma: no cover - wandb is optional
    wandb = None


ArrayLike = np.ndarray


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 10
    loss: str = "cross_entropy"
    use_wandb: bool = False
    wandb_frequency: int = 1


def train_model(
    model,
    optimizer,
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: Optional[ArrayLike] = None,
    y_val: Optional[ArrayLike] = None,
    config: TrainingConfig | None = None,
) -> Dict[str, list]:
    """Train the model and return history metrics."""

    cfg = config or TrainingConfig()
    loss_fn, _ = get_loss(cfg.loss)

    num_classes = model.config.output_size  # type: ignore[attr-defined]
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes) if y_val is not None else None

    history: Dict[str, list] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    wandb_run = wandb.run if (cfg.use_wandb and wandb is not None) else None

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for x_batch, y_batch in create_batches(x_train, y_train_onehot, cfg.batch_size):
            y_pred = model.forward(x_batch)
            loss = loss_fn(y_batch, y_pred)

            if model.config.l2_coeff > 0:  # type: ignore[attr-defined]
                loss += l2_regularization(tuple(model.weights), model.config.l2_coeff)  # type: ignore[attr-defined]

            weight_grads, bias_grads = model.backward(y_batch, cfg.loss)
            model.update_parameters(weight_grads, bias_grads, optimizer)

            epoch_loss += loss
            epoch_acc += accuracy(y_batch, y_pred)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(avg_acc)

        if x_val is not None and y_val_onehot is not None:
            y_val_pred = model.forward(x_val)
            val_loss = loss_fn(y_val_onehot, y_val_pred)
            if model.config.l2_coeff > 0:  # type: ignore[attr-defined]
                val_loss += l2_regularization(tuple(model.weights), model.config.l2_coeff)  # type: ignore[attr-defined]
            val_acc = accuracy(y_val_onehot, y_val_pred)
        else:
            val_loss = float("nan")
            val_acc = float("nan")

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if wandb_run is not None and (epoch + 1) % cfg.wandb_frequency == 0:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} - "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return history


def evaluate_model(model, x_test: ArrayLike, y_test: ArrayLike, loss: str) -> Dict[str, float]:
    loss_fn, _ = get_loss(loss)
    num_classes = model.config.output_size  # type: ignore[attr-defined]
    y_test_onehot = one_hot_encode(y_test, num_classes)
    y_pred = model.forward(x_test)
    test_loss = loss_fn(y_test_onehot, y_pred)
    test_acc = accuracy(y_test_onehot, y_pred)
    return {"test_loss": test_loss, "test_accuracy": test_acc}
