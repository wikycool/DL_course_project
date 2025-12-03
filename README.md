# 02456 Deep Learning Assignment 1
### By Vignesh Sethuraman (s252755) and Sai Shashank Maktala (s253062)

This repository contains the implementation of a feedforward neural network from scratch using NumPy for the 02456 Deep Learning course at DTU.

## Project Structure

```
Deep_Learning_project/
├── src/                          # Source code package
│   ├── __init__.py              # Package initialization
│   ├── activations.py           # Activation functions (ReLU, Tanh, Sigmoid)
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── loss_functions.py        # Loss functions (cross-entropy, MSE, L2)
│   ├── metrics.py               # Evaluation metrics (accuracy)
│   ├── model.py                 # Feedforward neural network implementation
│   ├── optimizer.py             # Optimizers (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam)
│   └── trainer.py               # Training loop and evaluation
├── report/                       # Academic report
│   ├── report.tex               # LaTeX source
│   ├── report.pdf               # Compiled PDF
│   ├── confusion_matrix.png     # Best model confusion matrix
│   ├── 02456.sty                # LaTeX style file
│   └── IEEEbib.bst              # Bibliography style
├── experiments_demo.ipynb        # Jupyter notebook demonstrating key experiments
├── train.py                      # Command-line training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

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

- **Best Test Accuracy**: 82.59% on Fashion-MNIST (Nadam optimizer)
- **Best Configuration**: 
  - Optimizer: Nadam
  - Architecture: [128, 64]
  - Activation: Tanh (Xavier init)
  - Learning Rate: 0.001
  - No regularization
  - 20 epochs
- **Total Experiments**: 23 systematic runs tracked via Weights & Biases
- **Key Finding**: Optimizer selection dominates performance (42-point gap between Nadam and SGD)

## Datasets

- **Fashion-MNIST**: 60k training, 10k test images (28×28 grayscale)
- **CIFAR-10**: 50k training, 10k test images (32×32 RGB)

## Supervisor

- **Viswanathan Sankar** (viswa@dtu.dk)

## License

This project is for academic purposes as part of the 02456 Deep Learning course at DTU.
