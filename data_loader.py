"""
Data loading and preprocessing utilities for neural network training.
Supports MNIST, Fashion-MNIST, CIFAR-10, and custom datasets.
"""

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class DataLoader:
    """Handles loading and preprocessing of various datasets."""
    
    def __init__(self, dataset_name='mnist', data_dir='./data'):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset ('mnist', 'fashion_mnist', 'cifar10')
            data_dir: Directory to store/load data
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_data(self, normalize=True, flatten=True, one_hot=True):
        """
        Load and preprocess the specified dataset.
        
        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            flatten: Whether to flatten images to 1D vectors
            one_hot: Whether to one-hot encode labels
            
        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        print(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'mnist':
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            num_classes = 10
            
        elif self.dataset_name == 'fashion_mnist':
            (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
            num_classes = 10
            
        elif self.dataset_name == 'cifar10':
            (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
            y_train = y_train.flatten()
            y_test = y_test.flatten()
            num_classes = 10
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Normalize pixel values
        if normalize:
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            
        # Flatten images
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
            print(f"Images flattened to shape: {X_train.shape[1:]}")
        
        # One-hot encode labels
        if one_hot:
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            print(f"Labels one-hot encoded: {num_classes} classes")
        
        return (X_train, y_train), (X_test, y_test)
    
    def create_validation_split(self, X_train, y_train, val_size=0.2, random_state=42):
        """
        Split training data into train and validation sets.
        
        Args:
            X_train: Training features
            y_train: Training labels
            val_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=random_state,
            stratify=y_train if len(y_train.shape) == 1 else np.argmax(y_train, axis=1)
        )
        
        print(f"Validation split created: {X_train.shape[0]} train, {X_val.shape[0]} validation samples")
        return X_train, X_val, y_train, y_val
    
    def get_batch(self, X, y, batch_size, shuffle=True):
        """
        Generate batches for training.
        
        Args:
            X: Features
            y: Labels
            batch_size: Size of each batch
            shuffle: Whether to shuffle data before batching
            
        Yields:
            (X_batch, y_batch)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]


def load_custom_dataset(X_path, y_path, test_size=0.2, normalize=True, random_state=42):
    """
    Load a custom dataset from numpy files.
    
    Args:
        X_path: Path to features .npy file
        y_path: Path to labels .npy file
        test_size: Fraction of data to use for testing
        normalize: Whether to normalize features
        random_state: Random seed
        
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    X = np.load(X_path)
    y = np.load(y_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader('mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    print(f"\nDataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
