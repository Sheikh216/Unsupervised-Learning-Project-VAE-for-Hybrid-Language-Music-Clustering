"""
Loss functions for neural network training.
Includes forward pass and gradient computation.
"""

import numpy as np


class Loss:
    """Base class for loss functions."""
    
    def forward(self, y_pred, y_true):
        """Compute loss."""
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        """Compute gradient of loss."""
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean Squared Error loss for regression."""
    
    def forward(self, y_pred, y_true):
        """MSE: mean((y_pred - y_true)²)"""
        return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
    
    def backward(self, y_pred, y_true):
        """Gradient: 2 * (y_pred - y_true) / n"""
        return 2 * (y_pred - y_true) / y_true.shape[0]


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss for binary classification."""
    
    def forward(self, y_pred, y_true):
        """BCE: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred, y_true):
        """Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy loss for multi-class classification."""
    
    def forward(self, y_pred, y_true):
        """CCE: -mean(sum(y_true * log(y_pred)))"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_pred, y_true):
        """Gradient: (y_pred - y_true)"""
        # For softmax + cross-entropy, the gradient simplifies to (y_pred - y_true)
        return (y_pred - y_true) / y_true.shape[0]


class HuberLoss(Loss):
    """Huber loss - robust to outliers."""
    
    def __init__(self, delta=1.0):
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        """Huber loss: quadratic for small errors, linear for large errors"""
        error = y_pred - y_true
        is_small_error = np.abs(error) <= self.delta
        
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
    def backward(self, y_pred, y_true):
        """Gradient of Huber loss"""
        error = y_pred - y_true
        return np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error)
        ) / y_true.shape[0]


def get_loss(name):
    """
    Get loss function by name.
    
    Args:
        name: Name of loss function
        
    Returns:
        Loss function instance
    """
    losses = {
        'mse': MeanSquaredError(),
        'mean_squared_error': MeanSquaredError(),
        'binary_crossentropy': BinaryCrossEntropy(),
        'categorical_crossentropy': CategoricalCrossEntropy(),
        'cross_entropy': CategoricalCrossEntropy(),
        'huber': HuberLoss()
    }
    
    name = name.lower()
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name]
