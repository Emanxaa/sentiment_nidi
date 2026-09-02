"""
IndoBERTweet Metrics Calculation & Serialization Module
=======================================================
Computes evaluation metrics (Accuracy, Macro F1, Precision, Recall)
and classification reports for transformer models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

CLASS_NAMES_DEFAULT = ["Negative", "Neutral", "Positive"]


def compute_metrics(eval_pred) -> Dict[str, float]:
    """HuggingFace Trainer evaluation hook."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    precision = float(precision_score(labels, preds, average="macro", zero_division=0))
    recall = float(recall_score(labels, preds, average="macro", zero_division=0))
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def calculate_metrics_from_preds(
    y_true: Sequence[int],
    y_pred: Sequence[int]
) -> Dict[str, float]:
    """Calculate summary classification metrics from label sequences."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "macro_f1": round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4),
        "precision": round(float(precision_score(y_t, y_p, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_t, y_p, average="macro", zero_division=0)), 4),
    }


def generate_classification_report_df(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str] | None = None
) -> pd.DataFrame:
    """Generate detailed classification report DataFrame for all classes."""
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT
    labels = list(range(len(class_names)))
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(rep).transpose()
    df["support"] = df["support"].astype(int)
    return df
