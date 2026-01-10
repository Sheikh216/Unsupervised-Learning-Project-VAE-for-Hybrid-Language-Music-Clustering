"""
Main script to train neural networks on music/audio datasets.
Supports GTZAN, Million Song Dataset, Jamendo, and other music datasets.
"""

import numpy as np
import argparse
import os
from datetime import datetime

from audio_data_loader import AudioDataLoader, download_dataset_instructions
from neural_network import NeuralNetwork
from trainer import Trainer
from visualizer import Visualizer


def train_music_classifier(args):
    """Train a neural network on music/audio dataset."""
    print("\n" + "="*60)
    print(f"TRAINING NEURAL NETWORK ON {args.dataset.upper()} DATASET")
    print("="*60 + "\n")
    
    # Load audio data
    loader = AudioDataLoader(args.dataset, data_dir=args.data_dir)
    
    # Load dataset with optional file paths
    load_kwargs = {}
    if args.features_file:
        load_kwargs['features_file'] = args.features_file
    if args.csv_file:
        load_kwargs['csv_file'] = args.csv_file
    
    (X_train, y_train), (X_test, y_test) = loader.load_data(**load_kwargs)
    
    # Create validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]} dimensions")
    print(f"  Classes: {y_train.shape[1]}")
    
    # Create neural network
    print(f"\nCreating neural network with architecture: {args.architecture}")
    layer_sizes = [X_train.shape[1]] + args.architecture + [y_train.shape[1]]
    
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
        viz.plot_training_history(history, save=True, 
                                  filename=f'{args.dataset}_training_history.png')
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get genre names
        genre_names = get_genre_names(args.dataset, y_train.shape[1])
        
        # Plot confusion matrix
        viz.plot_confusion_matrix(
            y_test, y_pred,
            class_names=genre_names,
            save=True,
            filename=f'{args.dataset}_confusion_matrix.png'
        )
        
        # Print classification report
        viz.print_classification_report(
            y_test, y_pred,
            class_names=genre_names
        )
        
        # Plot weight distribution
        viz.plot_weight_distribution(model, save=True,
                                    filename=f'{args.dataset}_weights.png')
    
    # Save model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output_dir, f'{args.dataset}_model_{timestamp}.pkl')
        model.save(model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    return model, history


def get_genre_names(dataset, n_classes):
    """Get genre names for the dataset."""
    if dataset == 'gtzan':
        return ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop',
                'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    elif dataset in ['msd', 'million_song']:
        return [f'Genre {i+1}' for i in range(n_classes)]
    elif dataset == 'jamendo':
        return [f'Genre {i+1}' for i in range(n_classes)]
    else:
        return [f'Class {i}' for i in range(n_classes)]


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train neural networks on music/audio datasets'
    )
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='gtzan',
                       choices=['gtzan', 'msd', 'million_song', 'jamendo', 'mir1k'],
                       help='Dataset to use for training')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='./music_data',
                       help='Directory containing the datasets')
    parser.add_argument('--features-file', type=str, default=None,
                       help='Pre-computed features file (for GTZAN)')
    parser.add_argument('--csv-file', type=str, default=None,
                       help='CSV file with features (for MSD, Jamendo)')
    
    # Model architecture
    parser.add_argument('--architecture', type=int, nargs='+', 
                       default=[128, 64],
                       help='Hidden layer sizes (e.g., --architecture 128 64)')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for optimization')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='momentum',
                       choices=['sgd', 'momentum'],
                       help='Optimizer to use')
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true', default=True,
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./music_results',
                       help='Directory to save outputs')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations')
    
    # Help
    parser.add_argument('--show-download-info', action='store_true',
                       help='Show dataset download instructions')
    
    args = parser.parse_args()
    
    # Show download instructions if requested
    if args.show_download_info:
        download_dataset_instructions()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model, history = train_music_classifier(args)
    
    return model, history


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main function
    model, history = main()
