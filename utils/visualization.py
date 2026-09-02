"""
Visualization Module (Milestone M1)
===================================
Provides standardized plotting functions for:
- Loss curves (Train vs Validation)
- Accuracy curves (Train vs Validation)
- Uniform Confusion Matrices (Train, Val, Test) with labels [Negative, Neutral, Positive]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


DEFAULT_LABELS = ["Negative", "Neutral", "Positive"]


def plot_learning_curves(
    history_df: pd.DataFrame,
    loss_output_path: str | Path,
    acc_output_path: str | Path
) -> None:
    """
    Generate and save separate Loss and Accuracy learning curves.

    Args:
        history_df: DataFrame with columns 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        loss_output_path: Destination path for loss_curve.png.
        acc_output_path: Destination path for accuracy_curve.png.
    """
    loss_path = Path(loss_output_path)
    acc_path = Path(acc_output_path)
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    acc_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history_df["epoch"]

    # 1. Loss Curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history_df["train_loss"], marker="o", linewidth=2, color="#1f77b4", label="Train Loss")
    plt.plot(epochs, history_df["val_loss"], marker="s", linewidth=2, color="#ff7f0e", linestyle="--", label="Validation Loss")
    plt.title("LSTM Training & Validation Loss Curve", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history_df["train_acc"], marker="o", linewidth=2, color="#2ca02c", label="Train Accuracy")
    plt.plot(epochs, history_df["val_acc"], marker="s", linewidth=2, color="#d62728", linestyle="--", label="Validation Accuracy")
    plt.title("LSTM Training & Validation Accuracy Curve", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(epochs)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    output_path: str | Path,
    labels: List[str] | None = None,
    title: str = "Confusion Matrix"
) -> None:
    """
    Generate and save a uniformly styled confusion matrix heatmap.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted label values.
        output_path: Destination path for confusion matrix image.
        labels: Class label names in order [0, 1, 2]. Defaults to ['Negative', 'Neutral', 'Positive'].
        title: Plot title.
    """
    if labels is None:
        labels = DEFAULT_LABELS

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 13, "fontweight": "bold"}
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=11, labelpad=8)
    plt.ylabel("Actual Sentiment", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
