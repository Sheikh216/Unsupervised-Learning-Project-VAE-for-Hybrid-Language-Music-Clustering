"""
Example notebook demonstrating how to use the neural network library.
Run this as a standalone script to see the neural network in action.
"""

import numpy as np
from data_loader import DataLoader
from neural_network import NeuralNetwork
from trainer import Trainer
from visualizer import Visualizer


def example_1_simple_mnist():
    """Example 1: Simple MNIST training."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple MNIST Training")
    print("="*70 + "\n")
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    loader = DataLoader('mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    # Use a subset for faster training
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    print(f"Using {X_train.shape[0]} training samples")
    
    # Create a simple network: 784 -> 64 -> 10
    print("\nCreating neural network...")
    model = NeuralNetwork(
        layer_sizes=[784, 64, 10],
        activations=['relu', 'softmax'],
        loss='categorical_crossentropy'
    )
    
    # Train
    print("\nTraining model...")
    trainer = Trainer(model, learning_rate=0.1, batch_size=128, epochs=10)
    history = trainer.train(X_train, y_train, verbose=True)
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model, history


def example_2_deep_network():
    """Example 2: Deeper network with validation."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Deep Network with Validation")
    print("="*70 + "\n")
    
    # Load data
    loader = DataLoader('mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    # Create validation split
    X_train, X_val, y_train, y_val = loader.create_validation_split(
        X_train, y_train, val_size=0.2
    )
    
    # Use subset
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_val = X_val[:2000]
    y_val = y_val[:2000]
    
    # Create deeper network: 784 -> 128 -> 64 -> 32 -> 10
    print("\nCreating deep neural network...")
    model = NeuralNetwork(
        layer_sizes=[784, 128, 64, 32, 10],
        activations=['relu', 'relu', 'relu', 'softmax'],
        loss='categorical_crossentropy'
    )
    
    # Train with validation
    print("\nTraining model with validation...")
    trainer = Trainer(model, learning_rate=0.05, batch_size=64, epochs=15)
    history = trainer.train(
        X_train, y_train, 
        X_val, y_val,
        early_stopping=True,
        patience=3,
        verbose=True
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Visualize
    print("\nGenerating visualizations...")
    viz = Visualizer(save_dir='./examples_output')
    viz.plot_training_history(history, save=True, filename='example2_history.png')
    
    return model, history


def example_3_fashion_mnist():
    """Example 3: Fashion-MNIST classification."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Fashion-MNIST Classification")
    print("="*70 + "\n")
    
    # Load Fashion-MNIST
    loader = DataLoader('fashion_mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    # Use subset
    X_train = X_train[:8000]
    y_train = y_train[:8000]
    
    # Create network
    model = NeuralNetwork(
        layer_sizes=[784, 256, 128, 10],
        activations=['relu', 'relu', 'softmax'],
        loss='categorical_crossentropy'
    )
    
    # Train
    trainer = Trainer(model, learning_rate=0.05, batch_size=128, epochs=12)
    history = trainer.train(X_train, y_train, verbose=True)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Visualize predictions
    viz = Visualizer(save_dir='./examples_output')
    y_pred = model.predict(X_test)
    
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    viz.plot_sample_predictions(
        X_test, y_test, y_pred, 
        n_samples=15,
        save=True,
        filename='example3_predictions.png'
    )
    
    viz.print_classification_report(y_test, y_pred, class_names=class_names)
    
    return model, history


def example_4_optimizer_comparison():
    """Example 4: Compare different optimizers."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Optimizer Comparison (SGD vs Momentum)")
    print("="*70 + "\n")
    
    # Load data
    loader = DataLoader('mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    # Use subset for speed
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    # Train with SGD
    print("\nTraining with SGD...")
    model_sgd = NeuralNetwork(layer_sizes=[784, 128, 10])
    trainer_sgd = Trainer(model_sgd, learning_rate=0.05, optimizer='sgd', epochs=10)
    history_sgd = trainer_sgd.train(X_train, y_train, verbose=False)
    acc_sgd = model_sgd.evaluate(X_test, y_test)[1]
    
    # Train with Momentum
    print("Training with Momentum...")
    model_momentum = NeuralNetwork(layer_sizes=[784, 128, 10])
    trainer_momentum = Trainer(model_momentum, learning_rate=0.05, optimizer='momentum', epochs=10)
    history_momentum = trainer_momentum.train(X_train, y_train, verbose=False)
    acc_momentum = model_momentum.evaluate(X_test, y_test)[1]
    
    # Compare results
    print("\n" + "-"*70)
    print("COMPARISON RESULTS")
    print("-"*70)
    print(f"SGD Test Accuracy:      {acc_sgd:.4f} ({acc_sgd*100:.2f}%)")
    print(f"Momentum Test Accuracy: {acc_momentum:.4f} ({acc_momentum*100:.2f}%)")
    print("-"*70)
    
    # Visualize comparison
    from visualizer import plot_multiple_metrics
    
    histories = {
        'SGD': history_sgd,
        'Momentum': history_momentum
    }
    
    print("\nGenerating comparison plots...")
    plot_multiple_metrics(
        histories, 
        metric_name='loss',
        save_dir='./examples_output',
        filename='example4_loss_comparison.png'
    )
    
    plot_multiple_metrics(
        histories, 
        metric_name='acc',
        save_dir='./examples_output',
        filename='example4_acc_comparison.png'
    )
    
    return histories


def example_5_save_and_load():
    """Example 5: Save and load a trained model."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Save and Load Model")
    print("="*70 + "\n")
    
    # Load data
    loader = DataLoader('mnist')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    
    X_train = X_train[:3000]
    y_train = y_train[:3000]
    
    # Train a model
    print("Training model...")
    model = NeuralNetwork(layer_sizes=[784, 64, 10])
    trainer = Trainer(model, learning_rate=0.1, batch_size=128, epochs=5)
    trainer.train(X_train, y_train, verbose=False)
    
    # Evaluate before saving
    acc_before = model.evaluate(X_test, y_test)[1]
    print(f"\nAccuracy before saving: {acc_before:.4f}")
    
    # Save model
    model_path = './examples_output/example_model.pkl'
    model.save(model_path)
    
    # Create new model and load weights
    print("\nLoading model from disk...")
    new_model = NeuralNetwork(layer_sizes=[784, 64, 10])
    new_model.load(model_path)
    
    # Evaluate loaded model
    acc_after = new_model.evaluate(X_test, y_test)[1]
    print(f"Accuracy after loading: {acc_after:.4f}")
    
    # Verify they match
    if abs(acc_before - acc_after) < 1e-6:
        print("\n✓ Model successfully saved and loaded!")
    else:
        print("\n✗ Warning: Model accuracy changed after loading!")
    
    return model, new_model


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "NEURAL NETWORK EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates various features of the neural network library.")
    print("Note: Using small subsets of data for faster execution.\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run examples
        print("\n[1/5] Running Example 1...")
        example_1_simple_mnist()
        
        print("\n[2/5] Running Example 2...")
        example_2_deep_network()
        
        print("\n[3/5] Running Example 3...")
        example_3_fashion_mnist()
        
        print("\n[4/5] Running Example 4...")
        example_4_optimizer_comparison()
        
        print("\n[5/5] Running Example 5...")
        example_5_save_and_load()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nCheck the './examples_output' directory for visualizations.")
        print("\n")
        
    except Exception as e:
        print(f"\n\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
