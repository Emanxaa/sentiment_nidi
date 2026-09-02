"""
Evaluator Module (Milestone M1)
===============================
Provides modular evaluation functions for Train, Validation, and Test splits.
Computes Accuracy, Macro Precision, Macro Recall, and Macro F1.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import calculate_metrics


def evaluate_split(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities for a given DataLoader split.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for the target split.
        device: PyTorch device.

    Returns:
        Tuple of (y_true, y_pred, y_prob) as numpy arrays.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true_list.extend(batch_y.numpy())
            y_pred_list.extend(preds)
            y_prob_list.extend(probs)

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    y_prob = np.array(y_prob_list, dtype=np.float32)

    return y_true, y_pred, y_prob


def evaluate_all_splits(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device
) -> Dict[str, Dict]:
    """
    Run full evaluation across Train, Validation, and Test splits.

    Args:
        model: Trained PyTorch model.
        loaders: Dictionary containing DataLoader for 'train', 'val', and 'test'.
        device: PyTorch device.

    Returns:
        Dictionary mapping split names to their evaluation metrics and predictions.
    """
    results = {}

    for split_name, loader in loaders.items():
        y_true, y_pred, y_prob = evaluate_split(model, loader, device)
        metrics = calculate_metrics(y_true, y_pred)
        results[split_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    return results
