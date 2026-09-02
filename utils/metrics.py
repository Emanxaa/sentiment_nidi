"""
Metrics Module (Milestone M1)
=============================
Computes evaluation metrics, formats classification reports, and saves
JSON and CSV metric artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


CLASS_NAMES_DEFAULT = ["Negative", "Neutral", "Positive"]


def calculate_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int]
) -> Dict[str, float]:
    """
    Calculate summary classification metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted label values.

    Returns:
        Dictionary with accuracy, macro_f1, precision, and recall.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def generate_classification_report_df(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    class_names: List[str] | None = None
) -> pd.DataFrame:
    """
    Generate detailed classification report DataFrame for all classes.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted label values.
        class_names: Class labels in order [0, 1, 2].

    Returns:
        pd.DataFrame containing precision, recall, f1-score, support.
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report_dict).transpose()
    df_report["support"] = df_report["support"].astype(int)
    return df_report


def save_metrics_json(
    metrics_data: dict,
    filepath: str | Path
) -> None:
    """
    Save master metrics dictionary to JSON file.

    Args:
        metrics_data: Dictionary of experiment metrics.
        filepath: Output JSON path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)


def save_history_csv(
    history_df: pd.DataFrame,
    filepath: str | Path
) -> None:
    """
    Save training history per epoch to CSV.

    Args:
        history_df: DataFrame containing per-epoch loss and accuracy metrics.
        filepath: Output CSV path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(path, index=False)


def save_classification_report_csv(
    report_df: pd.DataFrame,
    filepath: str | Path
) -> None:
    """
    Save classification report DataFrame to CSV.

    Args:
        report_df: Classification report DataFrame.
        filepath: Output CSV path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(path, index=True)
