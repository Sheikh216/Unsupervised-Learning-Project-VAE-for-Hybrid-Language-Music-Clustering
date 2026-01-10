"""
Neural Network implementation from scratch using NumPy.
Supports multiple layers, activation functions, and optimizers.
"""

import numpy as np
from activations import get_activation
from losses import get_loss
import pickle


class Layer:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size, activation='relu', 
                 weight_init='he', use_bias=True):
        """
        Initialize a dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features (neurons)
            activation: Activation function name
            weight_init: Weight initialization method ('he', 'xavier', 'random')
            use_bias: Whether to use bias terms
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        
        # Initialize weights
        if weight_init == 'he':
            # He initialization for ReLU
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init == 'xavier':
            # Xavier initialization for sigmoid/tanh
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            # Random small weights
            self.weights = np.random.randn(input_size, output_size) * 0.01
        
        # Initialize bias
        if use_bias:
            self.bias = np.zeros((1, output_size))
        else:
            self.bias = None
        
        # Cache for backpropagation
        self.input_cache = None
        self.z_cache = None
        self.output_cache = None
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Output after activation of shape (batch_size, output_size)
        """
        self.input_cache = x
        
        # Linear transformation: z = xW + b
        self.z_cache = np.dot(x, self.weights)
        if self.use_bias:
            self.z_cache += self.bias
        
        # Apply activation function
        self.output_cache = self.activation.forward(self.z_cache)
        
        return self.output_cache
    
    def backward(self, dA):
        """
        Backward pass through the layer.
        
        Args:
            dA: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        # Gradient of activation function
        dZ = dA * self.activation.backward(self.z_cache)
        
        # Gradients of weights and bias
        m = self.input_cache.shape[0]
        self.dW = np.dot(self.input_cache.T, dZ) / m
        if self.use_bias:
            self.db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Gradient with respect to input
        dX = np.dot(dZ, self.weights.T)
        
        return dX


class NeuralNetwork:
    """Multi-layer neural network."""
    
    def __init__(self, layer_sizes, activations=None, loss='categorical_crossentropy',
                 weight_init='he'):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer (default: relu for hidden, softmax for output)
            loss: Loss function name
            weight_init: Weight initialization method
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.loss_fn = get_loss(loss)
        
        # Default activations: ReLU for hidden layers, softmax for output
        if activations is None:
            activations = ['relu'] * (self.num_layers - 1) + ['softmax']
        
        # Create layers
        self.layers = []
        for i in range(self.num_layers):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init
            )
            self.layers.append(layer)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def forward(self, x):
        """
        Forward pass through the entire network.
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Network output
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_pred, y_true):
        """
        Backward pass through the entire network.
        
        Args:
            y_pred: Predicted output
            y_true: True labels
        """
        # Gradient of loss with respect to output
        dA = self.loss_fn.backward(y_pred, y_true)
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def update_weights(self, learning_rate, optimizer='sgd', **optimizer_params):
        """
        Update weights using the specified optimizer.
        
        Args:
            learning_rate: Learning rate
            optimizer: Optimizer name ('sgd', 'momentum', 'adam')
            optimizer_params: Additional optimizer parameters
        """
        for layer in self.layers:
            if optimizer == 'sgd':
                layer.weights -= learning_rate * layer.dW
                if layer.use_bias:
                    layer.bias -= learning_rate * layer.db
            
            elif optimizer == 'momentum':
                # Momentum optimization (to be implemented)
                beta = optimizer_params.get('beta', 0.9)
                if not hasattr(layer, 'vW'):
                    layer.vW = np.zeros_like(layer.weights)
                    layer.vb = np.zeros_like(layer.bias) if layer.use_bias else None
                
                layer.vW = beta * layer.vW + (1 - beta) * layer.dW
                layer.weights -= learning_rate * layer.vW
                
                if layer.use_bias:
                    layer.vb = beta * layer.vb + (1 - beta) * layer.db
                    layer.bias -= learning_rate * layer.vb
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x: Input array
            
        Returns:
            Predictions
        """
        return self.forward(x)
    
    def evaluate(self, x, y):
        """
        Evaluate the model on given data.
        
        Args:
            x: Input features
            y: True labels
            
        Returns:
            (loss, accuracy)
        """
        y_pred = self.predict(x)
        loss = self.loss_fn.forward(y_pred, y)
        
        # Calculate accuracy
        if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
            y_true_classes = np.argmax(y, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_true_classes = y
            y_pred_classes = (y_pred > 0.5).astype(int)
        
        accuracy = np.mean(y_true_classes == y_pred_classes)
        
        return loss, accuracy
    
    def save(self, filepath):
        """Save model weights to a file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': [layer.weights for layer in self.layers],
            'biases': [layer.bias for layer in self.layers if layer.use_bias],
            'history': self.history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        for i, layer in enumerate(self.layers):
            layer.weights = model_data['weights'][i]
            if layer.use_bias:
                layer.bias = model_data['biases'][i]
        
        if 'history' in model_data:
            self.history = model_data['history']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test the neural network
    print("Testing Neural Network...")
    
    # Create a simple network: 784 -> 128 -> 64 -> 10
    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=['relu', 'relu', 'softmax'],
        loss='categorical_crossentropy'
    )
    
    # Test forward pass
    X_test = np.random.randn(32, 784)
    y_test = np.eye(10)[np.random.randint(0, 10, 32)]
    
    predictions = nn.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    loss, acc = nn.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
