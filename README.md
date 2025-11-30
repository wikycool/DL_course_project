# 02456 Deep Learning Assignment 1
### By Vignesh Sethuraman (s252755) and Sai Shashank Maktala (s253062)

This repository contains the implementation of a feedforward neural network from scratch using NumPy for the 02456 Deep Learning course at DTU.

## Project Structure

- **activations.py** - Activation functions (ReLU, Tanh, Sigmoid) and their derivatives
- **loss_functions.py** - Loss functions (cross-entropy, MSE) and L2 regularization
- **optimizer.py** - Optimization algorithms (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam)
- **model.py** - Feedforward neural network implementation with forward/backward propagation
- **data_loader.py** - Dataset loading and preprocessing utilities
- **trainer.py** - Training loop and evaluation functions
- **train.py** - Main training script with command-line interface
- **metrics.py** - Accuracy and evaluation metrics
- **report.tex** - Academic report documenting experiments and findings

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Train a model with default settings:

```bash
python train.py --dataset fashion_mnist --epochs 20 --optimizer adam --learning_rate 0.001
```

### Command-line Arguments

- `--dataset, -d`: Dataset to use (fashion_mnist, mnist, cifar10)
- `--epochs, -e`: Number of training epochs
- `--batch_size, -b`: Batch size for training
- `--optimizer, -o`: Optimizer (sgd, momentum, nesterov, rmsprop, adam, nadam)
- `--learning_rate, -lr`: Learning rate
- `--hidden_layers, -hl`: Hidden layer sizes (e.g., 256 128 64)
- `--activation, -a`: Activation function (relu, tanh, sigmoid)
- `--l2_coeff`: L2 regularization coefficient
- `--wandb_mode`: Weights & Biases logging mode (online, offline, disabled)

### Example Commands

```bash
# Train with Adam optimizer
python train.py -d fashion_mnist -e 20 -o adam -lr 0.001 -hl 256 128 64

# Train with different activation functions
python train.py -d fashion_mnist -e 20 -a relu -wi he

# Enable WandB tracking
python train.py -d fashion_mnist -e 20 --wandb_mode online
```

## Key Results

- **Best Test Accuracy**: 88.69% on Fashion-MNIST
- **Best Configuration**: Adam optimizer, [256, 128, 64] architecture, no L2 regularization
- **Total Experiments**: 23 systematic runs tracked via Weights & Biases

## Datasets

- **Fashion-MNIST**: 60k training, 10k test images (28×28 grayscale)
- **CIFAR-10**: 50k training, 10k test images (32×32 RGB)

## Supervisor

- **Viswanathan Sankar** (viswa@dtu.dk)

## License

This project is for academic purposes as part of the 02456 Deep Learning course at DTU.
