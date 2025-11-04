# Neural Network from Scratch with NumPy

This project implements a configurable fully-connected feedforward neural network (FFNN) using only NumPy. The refactored codebase integrates lessons learned from the `DL-assignment-01` reference solution while keeping the implementation lightweight and easy to extend.

## Key Features

- Modular `src/` package with clear separation of activations, models, optimizers, training utilities, and evaluation helpers
- Support for multiple optimizers (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam) and L2 regularisation
- Flexible data-loading pipeline leveraging either Keras datasets or local downloads
- Command line training script with optional Weights & Biases logging
- Comprehensive experiment notebook under `notebooks/` showing end-to-end workflows on Fashion-MNIST and CIFAR-10

## Installation

```bash
pip install -r requirements.txt
```

If you plan to log runs, authenticate with WandB:

```bash
wandb login
```

## Project Structure

```
Deep_Learning_project/
├── configs/
│   └── wandb_sweep.yaml      # WandB sweep configuration for hyperparameter search
├── notebooks/
│   └── main.ipynb            # Interactive experimentation notebook
├── scripts/
│   ├── train.py              # CLI entry-point for training and evaluation
│   ├── quick_test.py         # Smoke test for rapid pipeline verification
│   └── run_experiments.py    # Predefined experiment runner
├── src/
│   ├── __init__.py
│   ├── activations.py        # Activation functions and derivatives
│   ├── data/
│   │   └── loaders.py        # Dataset loaders, batching, one-hot helpers
│   ├── evaluation/
│   │   └── metrics.py        # Accuracy, confusion matrix, classification report
│   ├── models/
│   │   └── feedforward.py    # Feedforward network implementation
│   ├── optimizers.py         # Unified optimizer with multiple update rules
│   └── training/
│       └── trainer.py        # Training loop and evaluation helpers
├── requirements.txt          # Project dependencies
├── README.md                 # This document
├── assignment.txt            # Project brief
├── overview.txt              # Course overview
└── synopsis.md               # Submitted project synopsis
```

## Quick Start

Train a model from the command line:

```bash
python scripts/train.py \
    --dataset fashion_mnist \
    --hidden_layers 128 64 \
    --epochs 10 \
    --optimizer adam \
    --learning_rate 0.001 \
    --wandb_mode disabled
```

The script loads the dataset (via Keras by default), splits out a validation set, trains the network, and prints test metrics. Enable WandB logging with `--wandb_mode online` and optionally specify `--wandb_project` and `--wandb_entity`.

## Notebook Workflow

Open `notebooks/main.ipynb` for a step-by-step walkthrough covering:

- Dataset loading and preprocessing
- Model/optimizer configuration
- Training and validation curves
- Hyperparameter experiments (optimizers, activations)
- Evaluation with confusion matrices and classification reports

## Utility Scripts & Sweeps

- `scripts/quick_test.py` runs a two-epoch smoke test on a reduced dataset subset—useful after environment changes.
- `scripts/run_experiments.py` batches common experiment configurations (optimizers, activations, architectures). Run `python scripts/run_experiments.py --list` to inspect available setups.
- `configs/wandb_sweep.yaml` defines a WandB Bayesian sweep covering optimizers, activations, and regularisation strength. Launch with `wandb sweep configs/wandb_sweep.yaml`.

## Notes

- CIFAR-10 experiments are more computationally demanding; adjust hidden layer sizes, epochs, and batch size accordingly.

## Contributors

- Vignesh Sethuraman (`s252755@student.dtu.dk`)
- Sai Shashank Maktala (`s253062@student.dtu.dk`)

## Supervisor

Viswanathan Sankar (viswa@dtu.dk)

