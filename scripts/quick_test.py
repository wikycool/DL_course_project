"""Quick smoke test to verify the training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_dataset, train_val_split
from src.models.feedforward import FeedForwardNN, NetworkConfig
from src.optimizers import Optimizer, OptimizerConfig
from src.training.trainer import TrainingConfig, evaluate_model, train_model


def main(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    # Try keras first, fall back to local download
    try:
        x_train, y_train, x_test, y_test = load_dataset("fashion_mnist", source="keras")
    except ImportError:
        print("Keras not available, using local download...")
        x_train, y_train, x_test, y_test = load_dataset("fashion_mnist", source="local")

    # Use a small subset for speed
    subset = 1024
    indices = rng.choice(x_train.shape[0], subset, replace=False)
    x_subset = x_train[indices]
    y_subset = y_train[indices]

    x_train_small, y_train_small, x_val_small, y_val_small = train_val_split(x_subset, y_subset, validation_split=0.2)

    input_size = x_train_small.shape[1]
    output_size = int(np.max(y_train_small)) + 1

    network_config = NetworkConfig(
        input_size=input_size,
        hidden_sizes=[64],
        output_size=output_size,
        activation="relu",
        output_activation="softmax",
        weight_init="he",
        l2_coeff=1e-4,
    )

    optimizer_config = OptimizerConfig(
        optimizer_type="adam",
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    model = FeedForwardNN(network_config)
    optimizer = Optimizer(optimizer_config)
    train_config = TrainingConfig(batch_size=64, num_epochs=2, loss="cross_entropy", use_wandb=False)

    history = train_model(
        model,
        optimizer,
        x_train_small,
        y_train_small,
        x_val_small,
        y_val_small,
        config=train_config,
    )

    metrics = evaluate_model(model, x_test[:1000], y_test[:1000], loss="cross_entropy")

    print("Training loss history:", history["train_loss"])
    print("Validation loss history:", history["val_loss"])
    print("Test metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick training smoke test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(seed=args.seed)
