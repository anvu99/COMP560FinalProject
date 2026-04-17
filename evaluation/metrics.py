import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
import json
import os

def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Computes standard classification metrics for conflict detection.
    
    Args:
        y_true: Ground truth binary labels (1=conflict, 0=non-conflict).
        y_pred: Predicted binary labels.
        y_proba: (Optional) Predicted probabilities for the positive class.
        
    Returns:
        dict: Dictionary containing the computed metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0)
    }
    
    # Calculate AUROC if probabilities are provided
    if y_proba is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            # Handles cases where there's only one class in y_true
            metrics["auroc"] = float('nan')
            
    return metrics

def print_metrics(metrics_dict, method_name="Method"):
    """
    Prints a formatted metrics report.
    """
    print(f"\\n{'='*40}")
    print(f"Metrics Report: {method_name}")
    print(f"{'='*40}")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"{k.capitalize():<15}: {v:.4f}")
        else:
            print(f"{k.capitalize():<15}: {v}")
    print(f"{'='*40}\\n")

def save_results(results_dict, output_path):
    """
    Saves the results dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
