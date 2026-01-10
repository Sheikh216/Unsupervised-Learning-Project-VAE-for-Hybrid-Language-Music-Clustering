"""
Visualization utilities for neural network training.
Plots training curves, confusion matrices, and sample predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


class Visualizer:
    """Handles visualization of training results."""
    
    def __init__(self, save_dir='./plots'):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, save=True, filename='training_history.png'):
        """
        Plot training and validation loss and accuracy.
        
        Args:
            history: Training history dictionary
            save: Whether to save the plot
            filename: Filename to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        if 'val_acc' in history and history['val_acc']:
            axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                             save=True, filename='confusion_matrix.png'):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save: Whether to save the plot
            filename: Filename to save the plot
        """
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, X, y_true, y_pred, n_samples=10, 
                               image_shape=(28, 28), save=True,
                               filename='sample_predictions.png'):
        """
        Plot sample predictions with images.
        
        Args:
            X: Input images (flattened or not)
            y_true: True labels
            y_pred: Predicted labels
            n_samples: Number of samples to plot
            image_shape: Shape to reshape images to
            save: Whether to save the plot
            filename: Filename to save the plot
        """
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true_classes = np.argmax(y_true, axis=1)
        else:
            y_true_classes = y_true
        
        if len(y_pred.shape) > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        
        # Randomly select samples
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        # Plot
        n_cols = 5
        n_rows = int(np.ceil(n_samples / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
            
            # Reshape image if flattened
            img = X[idx].reshape(image_shape)
            
            # Determine if prediction is correct
            is_correct = y_true_classes[idx] == y_pred_classes[idx]
            color = 'green' if is_correct else 'red'
            
            # Plot image
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'True: {y_true_classes[idx]}\nPred: {y_pred_classes[idx]}',
                            color=color, fontsize=10)
        
        # Hide extra subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved to {save_path}")
        
        plt.show()
    
    def plot_learning_rate_schedule(self, schedule_fn, epochs, 
                                    save=True, filename='lr_schedule.png'):
        """
        Plot learning rate schedule.
        
        Args:
            schedule_fn: Learning rate schedule function
            epochs: Number of epochs
            save: Whether to save the plot
            filename: Filename to save the plot
        """
        lrs = [schedule_fn(epoch) for epoch in range(epochs)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning rate schedule saved to {save_path}")
        
        plt.show()
    
    def plot_weight_distribution(self, model, save=True, 
                                filename='weight_distribution.png'):
        """
        Plot distribution of weights in each layer.
        
        Args:
            model: Neural network model
            save: Whether to save the plot
            filename: Filename to save the plot
        """
        n_layers = len(model.layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, layer in enumerate(model.layers):
            weights = layer.weights.flatten()
            axes[i].hist(weights, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel('Weight Value', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'Layer {i+1} Weights\n(mean={weights.mean():.3f}, std={weights.std():.3f})',
                            fontsize=11)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Weight distribution saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self, y_true, y_pred, class_names=None):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
        """
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names))
        print("="*60 + "\n")


def plot_multiple_metrics(history_dict, metric_name='acc', save_dir='./plots',
                         save=True, filename='comparison.png'):
    """
    Plot comparison of a metric across multiple models.
    
    Args:
        history_dict: Dictionary of {model_name: history}
        metric_name: Metric to plot ('acc' or 'loss')
        save_dir: Directory to save plots
        save: Whether to save the plot
        filename: Filename to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, history in history_dict.items():
        train_key = f'train_{metric_name}'
        val_key = f'val_{metric_name}'
        
        if train_key in history:
            plt.plot(history[train_key], label=f'{model_name} (train)', linewidth=2)
        if val_key in history and history[val_key]:
            plt.plot(history[val_key], label=f'{model_name} (val)', 
                    linewidth=2, linestyle='--')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    plt.title(f'Model Comparison - {metric_name.capitalize()}', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualizer module loaded successfully!")
