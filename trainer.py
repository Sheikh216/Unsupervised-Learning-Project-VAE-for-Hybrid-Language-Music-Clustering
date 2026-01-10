"""
Training utilities for neural networks.
Includes training loop, validation, and early stopping.
"""

import numpy as np
from tqdm import tqdm
import time


class Trainer:
    """Handles training of neural networks."""
    
    def __init__(self, model, learning_rate=0.01, optimizer='sgd', 
                 batch_size=32, epochs=10):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            learning_rate: Learning rate for optimization
            optimizer: Optimizer name ('sgd', 'momentum', 'adam')
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Optimizer parameters
        self.optimizer_params = {}
        if optimizer == 'momentum':
            self.optimizer_params['beta'] = 0.9
        elif optimizer == 'adam':
            self.optimizer_params['beta1'] = 0.9
            self.optimizer_params['beta2'] = 0.999
            self.optimizer_params['epsilon'] = 1e-8
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              early_stopping=False, patience=5, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Training
            epoch_loss = 0
            epoch_acc = 0
            
            if verbose:
                batch_iterator = tqdm(range(n_batches), 
                                     desc=f"Epoch {epoch+1}/{self.epochs}")
            else:
                batch_iterator = range(n_batches)
            
            for batch_idx in batch_iterator:
                # Get batch
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.model.forward(X_batch)
                
                # Compute loss
                batch_loss = self.model.loss_fn.forward(y_pred, y_batch)
                epoch_loss += batch_loss
                
                # Compute accuracy
                if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
                    batch_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                else:
                    batch_acc = np.mean((y_pred > 0.5).astype(int) == y_batch)
                epoch_acc += batch_acc
                
                # Backward pass
                self.model.backward(y_pred, y_batch)
                
                # Update weights
                self.model.update_weights(
                    self.learning_rate, 
                    self.optimizer, 
                    **self.optimizer_params
                )
                
                if verbose:
                    batch_iterator.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'acc': f'{batch_acc:.4f}'
                    })
            
            # Average metrics over batches
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            # Store training metrics
            self.model.history['train_loss'].append(epoch_loss)
            self.model.history['train_acc'].append(epoch_acc)
            
            # Validation
            val_loss, val_acc = 0, 0
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.model.evaluate(X_val, y_val)
                self.model.history['val_loss'].append(val_loss)
                self.model.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                      f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}", end='')
                if X_val is not None:
                    print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print()
            
            # Early stopping
            if early_stopping and X_val is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        break
        
        return self.model.history
    
    def set_learning_rate(self, learning_rate):
        """Update learning rate."""
        self.learning_rate = learning_rate
    
    def get_learning_rate_schedule(self, schedule_type='step', **params):
        """
        Get a learning rate schedule function.
        
        Args:
            schedule_type: Type of schedule ('step', 'exponential', 'cosine')
            params: Schedule parameters
            
        Returns:
            Schedule function
        """
        if schedule_type == 'step':
            def step_schedule(epoch):
                drop_every = params.get('drop_every', 10)
                drop_rate = params.get('drop_rate', 0.5)
                return self.learning_rate * (drop_rate ** (epoch // drop_every))
            return step_schedule
        
        elif schedule_type == 'exponential':
            def exp_schedule(epoch):
                decay_rate = params.get('decay_rate', 0.95)
                return self.learning_rate * (decay_rate ** epoch)
            return exp_schedule
        
        elif schedule_type == 'cosine':
            def cosine_schedule(epoch):
                T_max = params.get('T_max', self.epochs)
                eta_min = params.get('eta_min', 0)
                return eta_min + (self.learning_rate - eta_min) * \
                       (1 + np.cos(np.pi * epoch / T_max)) / 2
            return cosine_schedule
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Neural network model
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights:
                # Save best weights
                self.best_weights = [layer.weights.copy() for layer in model.layers]
                self.best_biases = [layer.bias.copy() if layer.use_bias else None 
                                   for layer in model.layers]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
                if self.restore_best_weights and self.best_weights is not None:
                    # Restore best weights
                    for i, layer in enumerate(model.layers):
                        layer.weights = self.best_weights[i]
                        if layer.use_bias:
                            layer.bias = self.best_biases[i]
                    print(f"Restored best weights from epoch {self.patience} ago")
        
        return self.should_stop


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
