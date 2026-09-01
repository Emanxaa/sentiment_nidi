"""Data: loader dataset fleksibel (path mount Kaggle CLI 2.x vs lama), split, dataset PyTorch.

Sumber kebenaran loading data untuk semua eksperimen. Sel notebook menyuntik source
fungsi-fungsi di sini (via generator) sehingga notebook tetap self-contained di Kaggle.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

COL_TEXT = "text_bert"
COL_LABEL = "label"
CSV_NAME = "data_preprocessed_with_emoticon.csv"


def find_dataset_csv() -> str:
    """Cari CSV dataset di /kaggle/input (path mount berubah antara CLI 2.x dan lama).

    - CLI 2.x:  /kaggle/input/datasets/<owner>/<slug>/...
    - Skema lama: /kaggle/input/<slug>/...
    Tidak ada hardcode path: cari dari daftar file ter-mount.
    """
    mounted = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == CSV_NAME:
                mounted.append(os.path.join(root, f))
    if not mounted:
        raise FileNotFoundError(
            f"Dataset '{CSV_NAME}' tidak ditemukan di /kaggle/input. "
            "Cek dataset_sources di kernel-metadata.json."
        )
    return mounted[0]


def load_dataframe() -> pd.DataFrame:
    """Muat CSV dataset dengan validasi kolom BERT eksplisit (text_bert)."""
    path = find_dataset_csv()
    print("CSV ditemukan di:", path)
    df = pd.read_csv(path)
    if COL_TEXT not in df.columns:
        raise ValueError(
            f"Kolom '{COL_TEXT}' tidak ditemukan di CSV. Kolom tersedia: {df.columns.tolist()}"
        )
    df[COL_TEXT] = df[COL_TEXT].fillna("").astype(str)
    print(f"Kolom BERT terpilih: {COL_TEXT} | Total baris: {len(df)}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Split 80:20 (test) lalu 90:10 (val) — protokol konsisten semua eksperimen."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[COL_LABEL]
    )
    X_train = train_df[COL_TEXT].values
    X_test = test_df[COL_TEXT].values
    y_train = train_df[COL_LABEL].values
    y_test = test_df[COL_LABEL].values

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )
    return {
        "X_train": X_train_final,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_final,
        "y_val": y_val,
        "y_test": y_test,
    }


class SentimenDataset(Dataset):
    """Dataset PyTorch numpy-friendly (menerima numpy array & pandas Series)."""

    def __init__(
        self,
        texts: Any,
        labels: Any,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts.values if isinstance(texts, pd.Series) else np.array(texts)
        self.labels = labels.values if isinstance(labels, pd.Series) else np.array(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class MLMDataset(Dataset):
    """Dataset PyTorch untuk Masked Language Modeling (TAPT)."""

    def __init__(self, texts: Any, tokenizer, max_length: int = 128):
        self.texts = texts.values if isinstance(texts, pd.Series) else np.array(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
