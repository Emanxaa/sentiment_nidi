"""Metrik evaluasi: compute_metrics (HF) + softmax numpy (untuk simpan probabilitas)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL_NAMES = ["negatif", "netral", "positif"]


def compute_metrics(eval_pred):
    """Metrik untuk HF Trainer (average='macro', zero_division=0)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Softmax stabil (numerik) di axis=1."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prediction_frame(
    texts,
    y_true,
    logits: np.ndarray,
    prob_cols=("prob_negatif", "prob_netral", "prob_positif"),
):
    """DataFrame per-sampel: teks, label aktual/prediksi, probabilitas per kelas."""
    import pandas as pd

    y_pred = np.argmax(logits, axis=1)
    P = softmax_np(logits)
    return pd.DataFrame(
        {
            "text": pd.Series(texts),
            "label_aktual": pd.Series(y_true),
            "label_prediksi": pd.Series(y_pred),
            prob_cols[0]: P[:, 0],
            prob_cols[1]: P[:, 1],
            prob_cols[2]: P[:, 2],
        }
    )
