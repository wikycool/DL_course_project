"""Run predefined experiment configurations using the training CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train import main as train_main


ExperimentConfig = Dict[str, Sequence[int] | str | int | float]


EXPERIMENTS: Dict[str, List[ExperimentConfig]] = {
    "optimizers": [
        {
            "dataset": "fashion_mnist",
            "hidden_layers": [128, 64],
            "optimizer": opt,
            "learning_rate": 1e-3,
            "epochs": 12,
            "batch_size": 64,
            "activation": "relu",
            "wandb_mode": "disabled",
        }
        for opt in ["sgd", "momentum", "adam", "rmsprop"]
    ],
    "activations": [
        {
            "dataset": "fashion_mnist",
            "hidden_layers": [256, 128],
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "epochs": 15,
            "batch_size": 64,
            "activation": act,
            "weight_init": "he" if act == "relu" else "xavier",
            "wandb_mode": "disabled",
        }
        for act in ["relu", "tanh", "sigmoid"]
    ],
    "architectures": [
        {
            "dataset": "fashion_mnist",
            "hidden_layers": architecture,
            "optimizer": "adam",
            "learning_rate": 8e-4,
            "epochs": 15,
            "batch_size": 64,
            "activation": "relu",
            "wandb_mode": "disabled",
        }
        for architecture in ([128, 64], [256, 128, 64], [512, 256, 128])
    ],
}


def config_to_args(config: ExperimentConfig) -> List[str]:
    args: List[str] = []
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])
    return args


def run_experiment_group(name: str, limit: int | None = None) -> None:
    group = EXPERIMENTS.get(name)
    if not group:
        raise ValueError(f"Unknown experiment group '{name}'. Available: {list(EXPERIMENTS)}")

    selected = group[:limit] if limit else group
    for idx, config in enumerate(selected, start=1):
        args = config_to_args(config)
        print(f"\n>>> Running {name} experiment #{idx} with args: {' '.join(args)}")
        train_main(args)


def list_experiments() -> None:
    for name, configs in EXPERIMENTS.items():
        print(f"\n{name} ({len(configs)} runs)")
        for idx, config in enumerate(configs, start=1):
            printable = {k: config[k] for k in sorted(config.keys())}
            print(f"  [{idx}] {printable}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predefined experiment configurations.")
    parser.add_argument(
        "--group",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment group to run. Use --list to see available groups.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally run only the first N configurations in the group.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiment configurations without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list or not args.group:
        list_experiments()
        if not args.group:
            return

    run_experiment_group(args.group, limit=args.limit)


if __name__ == "__main__":
    main()
