"""
IndoBERTweet Multi-Split Evaluator Module
========================================
Runs deterministic evaluation across Train, Val, and Test sets
using the restored best checkpoint.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
from transformers import Trainer

from utils.bert_metrics import calculate_metrics_from_preds


def evaluate_all_splits(
    trainer: Trainer,
    train_dataset,
    val_dataset,
    test_dataset
) -> Dict[str, dict]:
    """
    Perform multi-split predictions and compute classification metrics.
    """
    train_pred = trainer.predict(train_dataset)
    val_pred = trainer.predict(val_dataset)
    test_pred = trainer.predict(test_dataset)

    y_train_pred = np.argmax(train_pred.predictions, axis=1)
    y_val_pred = np.argmax(val_pred.predictions, axis=1)
    y_test_pred = np.argmax(test_pred.predictions, axis=1)

    return {
        "train": {
            "metrics": calculate_metrics_from_preds(train_dataset.labels, y_train_pred),
            "y_true": train_dataset.labels,
            "y_pred": y_train_pred.tolist(),
        },
        "val": {
            "metrics": calculate_metrics_from_preds(val_dataset.labels, y_val_pred),
            "y_true": val_dataset.labels,
            "y_pred": y_val_pred.tolist(),
        },
        "test": {
            "metrics": calculate_metrics_from_preds(test_dataset.labels, y_test_pred),
            "y_true": test_dataset.labels,
            "y_pred": y_test_pred.tolist(),
        },
    }
