"""
Utilities for model evaluation and metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve
)


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return accuracy_score(y_true, y_pred)


def calculate_precision_recall_f1(y_true, y_pred, average='weighted'):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class
        
    Returns:
        (precision, recall, f1)
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return precision, recall, f1


def calculate_per_class_metrics(y_true, y_pred, num_classes=10):
    """
    Calculate metrics for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of per-class metrics
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    metrics = {}
    
    for i in range(num_classes):
        # Binary classification for each class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'class_{i}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(y_true_binary)
        }
    
    return metrics


def print_metrics_summary(y_true, y_pred, class_names=None):
    """
    Print a comprehensive summary of metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    # Overall metrics
    accuracy = calculate_accuracy(y_true, y_pred)
    precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
    
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*60 + "\n")
    
    # Per-class metrics
    if len(y_true.shape) > 1:
        num_classes = y_true.shape[1]
    else:
        num_classes = len(np.unique(y_true))
    
    per_class = calculate_per_class_metrics(y_true, y_pred, num_classes)
    
    print("PER-CLASS METRICS")
    print("="*60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-"*60)
    
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        metrics = per_class[f'class_{i}']
        print(f"{class_name:<15} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} "
              f"{metrics['support']:<10.0f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Metrics module loaded successfully!")
