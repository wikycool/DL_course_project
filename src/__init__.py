"""
Deep Learning Project - Source Code
====================================

This package contains the implementation of a feedforward neural network from scratch.

Modules:
- activations: Activation functions (ReLU, Tanh, Sigmoid, Softmax)
- data_loader: Dataset loading and preprocessing utilities
- loss_functions: Loss functions (Cross-entropy, MSE)
- metrics: Evaluation metrics (Accuracy)
- model: Feedforward neural network implementation
- optimizer: Optimization algorithms (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam)
- trainer: Training and evaluation utilities
"""

__version__ = "1.0.0"
__author__ = "Vignesh Sethuraman, Sai Shashank Maktala"

# Import main classes for convenience
from .model import FeedForwardNN, NetworkConfig
from .optimizer import Optimizer, OptimizerConfig
from .trainer import TrainingConfig, train_model, evaluate_model
from .data_loader import load_dataset, train_val_split

__all__ = [
    'FeedForwardNN',
    'NetworkConfig',
    'Optimizer',
    'OptimizerConfig',
    'TrainingConfig',
    'train_model',
    'evaluate_model',
    'load_dataset',
    'train_val_split',
]
