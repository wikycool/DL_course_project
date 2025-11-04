# Project Synopsis: Implementing a Neural Network from Scratch with NumPy

## Project Title
Implementing a Neural Network from Scratch with NumPy: Training, Optimization, and Experiment Tracking with Weights & Biases (WandB)

## Contributors
- Vignesh Sethuraman (`s252755@student.dtu.dk`)
- Sai Shashank Maktala (`s253062@student.dtu.dk`)

## Supervisor
Viswanathan Sankar (viswa@dtu.dk)

## Motivation
Understanding the fundamental mechanisms of neural networks is crucial for deep learning practitioners. While modern frameworks like TensorFlow and PyTorch abstract away implementation details, building a neural network from scratch using only NumPy provides deep insights into forward propagation, backpropagation, gradient computation, and optimization algorithms. This project aims to demystify these core concepts by implementing a fully functional feedforward neural network without high-level deep learning libraries, enabling a thorough understanding of how neural networks learn and generalize.

## Background
Neural networks are computational models inspired by biological neural networks. The mathematical foundation relies on matrix operations, activation functions, and gradient-based optimization. A feedforward neural network (FFN) consists of layers of interconnected neurons, where each connection has a weight that gets adjusted during training through backpropagation. The training process involves computing the gradient of a loss function with respect to the network parameters and updating weights using optimization algorithms like gradient descent or its variants (e.g., Adam, RMSprop).

This project will implement these components manually: forward propagation computes predictions through weighted sums and activation functions, while backward propagation computes gradients using the chain rule. The implementation will support configurable hyperparameters including network architecture (number of layers and hidden units), learning rate, batch size, optimizer choice, activation functions, and regularization techniques. Evaluation will be performed on standard benchmarks (Fashion-MNIST and CIFAR-10) to demonstrate the network's ability to learn complex patterns and identify challenges such as overfitting.

## Implementation Details

### Core Components Implemented
- **FFNN Class**: Fully configurable feedforward neural network with support for arbitrary architectures
- **Forward Propagation**: Matrix multiplications with activation functions (ReLU, sigmoid, tanh, identity)
- **Backward Propagation**: Manual gradient computation using chain rule for all layers
- **Loss Functions**: Cross-entropy and mean squared error with L2 regularization support
- **Optimizers**: SGD, Momentum, Nesterov, RMSprop, Adam, and Nadam optimizers
- **Weight Initialization**: Xavier/Glorot, He initialization, and random initialization
- **Evaluation Metrics**: Accuracy, confusion matrix, and classification reports

### Datasets
- **Fashion-MNIST**: Primary dataset for experimentation (28×28 grayscale images, 10 classes)
- **CIFAR-10**: Secondary dataset for evaluation (32×32×3 RGB images, 10 classes)
- Data loading supports both Keras API and local download for offline use

### Experiment Tracking
- **Weights & Biases Integration**: Complete logging of training metrics, hyperparameters, and model artifacts
- **Hyperparameter Sweeps**: Bayesian optimization sweep configuration for systematic hyperparameter search
- **Visualization**: Learning curves, parameter histograms, gradient norms, and confusion matrices

## Milestones

### Week 1: Core Implementation ✓
- ✓ Implemented FFNN class structure with configurable hyperparameters
- ✓ Implemented forward pass with matrix operations and activation functions (ReLU, sigmoid, tanh, identity)
- ✓ Implemented loss computation (MSE and cross-entropy) with L2 regularization
- ✓ Implemented backward pass with manual gradient calculation using chain rule
- ✓ Verified correctness through testing

### Week 2: Training and Optimization ✓
- ✓ Implemented training loop with mini-batch gradient descent
- ✓ Integrated multiple optimizers (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam)
- ✓ Implemented weight initialization strategies (Xavier, He, random)
- ✓ Added evaluation metrics (accuracy, confusion matrix)
- ✓ Initial experiments on Fashion-MNIST dataset

### Week 3: Experimentation and Logging ✓
- ✓ Set up Weights & Biases (WandB) integration for experiment tracking
- ✓ Implemented logging for learning curves, parameter histograms, and gradient norms
- ✓ Created hyperparameter sweep configuration for systematic search
- ✓ Prepared experiment framework for different activation functions and initializations
- ✓ CIFAR-10 dataset support implemented
- ✓ L2 regularization implemented and configurable

### Week 4: Analysis and Documentation
- Compare performance across different configurations
- Generate summary reports and visualizations
- Document findings and insights
- Prepare final report and ensure code reproducibility

## Technical Implementation

The implementation is organized into a modular structure with separate components for activations, losses, optimizers, data loading, and model architecture. The codebase supports both command-line training scripts and interactive Jupyter notebooks for experimentation. All code is written from scratch using only NumPy, with no dependencies on high-level deep learning frameworks. The project includes comprehensive experiment management tools, hyperparameter sweep configurations, and integration with Weights & Biases for experiment tracking and visualization.

**Code Repository**: The complete implementation, including all source code, training scripts, experiment configurations, and documentation, is available in the project repository.

## References
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Nielsen, M. A. (2015). Neural Networks and Deep Learning. Determination Press.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the thirteenth international conference on artificial intelligence and statistics.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.
- Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

