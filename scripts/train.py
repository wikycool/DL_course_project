"""Command line training entry-point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_dataset, train_val_split
from src.models.feedforward import FeedForwardNN, NetworkConfig
from src.optimizers import Optimizer, OptimizerConfig
from src.training.trainer import TrainingConfig, evaluate_model, train_model

try:  # Optional dependency
    import wandb  # type: ignore
except Exception:  # pragma: no cover - wandb is optional
    wandb = None


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy-based feedforward neural network.")

    # WandB configuration
    parser.add_argument("--wandb_project", "-wp", default="numpy-ffnn", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", "-we", default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")

    # Dataset
    parser.add_argument("--dataset", "-d", choices=["fashion_mnist", "mnist", "cifar10"], default="fashion_mnist")
    parser.add_argument("--data_source", choices=["keras", "local"], default="keras")
    parser.add_argument("--validation_split", type=float, default=0.1)

    # Model architecture
    parser.add_argument("--hidden_layers", "-hl", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--activation", "-a", default="tanh", choices=["identity", "sigmoid", "tanh", "relu"])
    parser.add_argument("--output_activation", default="softmax")
    parser.add_argument("--weight_init", "-wi", default="xavier", choices=["random", "xavier", "he"])
    parser.add_argument("--l2_coeff", type=float, default=0.0)

    # Training hyper-parameters
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--loss", "-l", choices=["cross_entropy", "mse", "mean_squared_error"], default="cross_entropy")

    # Optimizer configuration
    parser.add_argument("--optimizer", "-o", default="adam", choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"])
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--momentum", "-m", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", "-eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0)

    return parser.parse_args(args=args)


def main(cli_args: Sequence[str] | None = None) -> None:
    args = parse_args(cli_args)

    # Load data - try keras first, fall back to local if needed
    try:
        x_train, y_train, x_test, y_test = load_dataset(
            dataset=args.dataset,
            source=args.data_source,
            normalize=True,
            flatten=True,
        )
    except ImportError:
        if args.data_source == "keras":
            print("Keras not available, falling back to local download...")
            x_train, y_train, x_test, y_test = load_dataset(
                dataset=args.dataset,
                source="local",
                normalize=True,
                flatten=True,
            )
        else:
            raise

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, validation_split=args.validation_split)

    input_size = x_train.shape[1]
    output_size = int(np.max(y_train)) + 1

    net_config = NetworkConfig(
        input_size=input_size,
        hidden_sizes=args.hidden_layers,
        output_size=output_size,
        activation=args.activation,
        output_activation=args.output_activation,
        weight_init=args.weight_init,
        l2_coeff=args.l2_coeff,
    )

    model = FeedForwardNN(net_config)

    opt_config = OptimizerConfig(
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
    )
    optimizer = Optimizer(opt_config)

    wandb_run = None
    if wandb is not None and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            mode=args.wandb_mode,
        )
        run_name = (
            f"hl{'-'.join(map(str, args.hidden_layers))}_"
            f"bs{args.batch_size}_lr{args.learning_rate}_opt{args.optimizer}"
        )
        wandb_run.name = run_name

    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        loss=args.loss,
        use_wandb=wandb_run is not None,
    )

    history = train_model(
        model,
        optimizer,
        x_train,
        y_train,
        x_val,
        y_val,
        config=train_cfg,
    )

    # Evaluate on the test set
    metrics = evaluate_model(model, x_test, y_test, loss=args.loss)
    y_test_onehot = np.eye(output_size)[y_test]
    y_test_pred = model.predict_proba(x_test)
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"],
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=y_test_pred,
                y_true=y_test,
                class_names=[str(i) for i in range(output_size)],
            ),
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
