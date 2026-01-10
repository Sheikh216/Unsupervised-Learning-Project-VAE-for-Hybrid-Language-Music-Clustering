"""
Main script to train and evaluate neural networks on various datasets.
Demonstrates the complete workflow from data loading to evaluation.
"""

import numpy as np
import argparse
import os
from datetime import datetime

from data_loader import DataLoader
from neural_network import NeuralNetwork
from trainer import Trainer
from visualizer import Visualizer


def train_mnist(args):
    """Train a neural network on MNIST dataset."""
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK ON MNIST")
    print("="*60 + "\n")
    
    # Load data
    loader = DataLoader('mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data(
        normalize=True, 
        flatten=True, 
        one_hot=True
    )
    
    # Create validation split
    X_train, X_val, y_train, y_val = loader.create_validation_split(
        X_train, y_train, 
        val_size=0.2
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Create neural network
    print(f"\nCreating neural network with architecture: {args.architecture}")
    layer_sizes = [X_train.shape[1]] + args.architecture + [10]
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        loss='categorical_crossentropy',
        weight_init='he'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Train model
    print(f"\nTraining model...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Optimizer: {args.optimizer}\n")
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        early_stopping=args.early_stopping,
        patience=args.patience,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Visualize results
    if args.visualize:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        viz = Visualizer(save_dir=args.output_dir)
        
        # Plot training history
        viz.plot_training_history(history, save=True)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot confusion matrix
        viz.plot_confusion_matrix(
            y_test, y_pred,
            class_names=[str(i) for i in range(10)],
            save=True
        )
        
        # Plot sample predictions
        viz.plot_sample_predictions(
            X_test, y_test, y_pred,
            n_samples=10,
            image_shape=(28, 28),
            save=True
        )
        
        # Print classification report
        viz.print_classification_report(
            y_test, y_pred,
            class_names=[str(i) for i in range(10)]
        )
        
        # Plot weight distribution
        viz.plot_weight_distribution(model, save=True)
    
    # Save model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output_dir, f'mnist_model_{timestamp}.pkl')
        model.save(model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    return model, history


def train_fashion_mnist(args):
    """Train a neural network on Fashion-MNIST dataset."""
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK ON FASHION-MNIST")
    print("="*60 + "\n")
    
    # Fashion-MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Load data
    loader = DataLoader('fashion_mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data(
        normalize=True, 
        flatten=True, 
        one_hot=True
    )
    
    # Create validation split
    X_train, X_val, y_train, y_val = loader.create_validation_split(
        X_train, y_train, 
        val_size=0.2
    )
    
    # Create neural network
    layer_sizes = [X_train.shape[1]] + args.architecture + [10]
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        loss='categorical_crossentropy',
        weight_init='he'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Train model
    print(f"\nTraining model...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        early_stopping=args.early_stopping,
        patience=args.patience,
        verbose=True
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Visualize results
    if args.visualize:
        viz = Visualizer(save_dir=args.output_dir)
        viz.plot_training_history(history, save=True, 
                                  filename='fashion_mnist_history.png')
        
        y_pred = model.predict(X_test)
        viz.plot_confusion_matrix(y_test, y_pred, class_names=class_names,
                                 save=True, filename='fashion_mnist_cm.png')
        viz.print_classification_report(y_test, y_pred, class_names=class_names)
    
    # Save model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output_dir, f'fashion_mnist_model_{timestamp}.pkl')
        model.save(model_path)
    
    return model, history


def train_cifar10(args):
    """Train a neural network on CIFAR-10 dataset."""
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK ON CIFAR-10")
    print("="*60 + "\n")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load data
    loader = DataLoader('cifar10')
    (X_train, y_train), (X_test, y_test) = loader.load_data(
        normalize=True, 
        flatten=True, 
        one_hot=True
    )
    
    # Create validation split
    X_train, X_val, y_train, y_val = loader.create_validation_split(
        X_train, y_train, 
        val_size=0.2
    )
    
    # Create neural network (larger for CIFAR-10)
    layer_sizes = [X_train.shape[1]] + args.architecture + [10]
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        loss='categorical_crossentropy',
        weight_init='he'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Train model
    print(f"\nTraining model...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        early_stopping=args.early_stopping,
        patience=args.patience,
        verbose=True
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Visualize results
    if args.visualize:
        viz = Visualizer(save_dir=args.output_dir)
        viz.plot_training_history(history, save=True, 
                                  filename='cifar10_history.png')
        
        y_pred = model.predict(X_test)
        viz.plot_confusion_matrix(y_test, y_pred, class_names=class_names,
                                 save=True, filename='cifar10_cm.png')
        viz.print_classification_report(y_test, y_pred, class_names=class_names)
    
    # Save model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output_dir, f'cifar10_model_{timestamp}.pkl')
        model.save(model_path)
    
    return model, history


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train neural networks on various datasets'
    )
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist', 'cifar10'],
                       help='Dataset to use for training')
    
    # Model architecture
    parser.add_argument('--architecture', type=int, nargs='+', 
                       default=[128, 64],
                       help='Hidden layer sizes (e.g., --architecture 128 64)')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for optimization')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'momentum'],
                       help='Optimizer to use')
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for early stopping')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save outputs')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train on selected dataset
    if args.dataset == 'mnist':
        model, history = train_mnist(args)
    elif args.dataset == 'fashion_mnist':
        model, history = train_fashion_mnist(args)
    elif args.dataset == 'cifar10':
        model, history = train_cifar10(args)
    
    return model, history


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main function
    model, history = main()
