"""
Activation functions for neural networks.
Includes forward pass and gradient computation for backpropagation.
"""

import numpy as np


class Activation:
    """Base class for activation functions."""
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self, x):
        """Compute gradient for backpropagation."""
        raise NotImplementedError


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, x):
        """Sigmoid: 1 / (1 + exp(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, x):
        """Gradient: sigmoid(x) * (1 - sigmoid(x))"""
        s = self.forward(x)
        return s * (1 - s)


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, x):
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    def backward(self, x):
        """Gradient: 1 if x > 0, else 0"""
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    """Leaky ReLU activation function."""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        """LeakyReLU: max(alpha * x, x)"""
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        """Gradient: 1 if x > 0, else alpha"""
        return np.where(x > 0, 1, self.alpha)


class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    
    def forward(self, x):
        """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return np.tanh(x)
    
    def backward(self, x):
        """Gradient: 1 - tanh²(x)"""
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    """Softmax activation function for multi-class classification."""
    
    def forward(self, x):
        """Softmax: exp(x) / sum(exp(x))"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x):
        """
        Gradient: For softmax with cross-entropy, 
        the gradient is computed in the loss function.
        """
        # This is typically computed with cross-entropy loss
        return np.ones_like(x)


class Linear(Activation):
    """Linear (identity) activation function."""
    
    def forward(self, x):
        """Linear: f(x) = x"""
        return x
    
    def backward(self, x):
        """Gradient: 1"""
        return np.ones_like(x)


def get_activation(name):
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function instance
    """
    activations = {
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'leaky_relu': LeakyReLU(),
        'tanh': Tanh(),
        'softmax': Softmax(),
        'linear': Linear()
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name]
