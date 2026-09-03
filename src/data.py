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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

COL_TEXT = "text_bert"
COL_LABEL = "label"
CSV_NAME = "data_preprocessed_with_emoticon.csv"


def find_dataset_csv(csv_name: str = CSV_NAME) -> str:
    """Cari CSV dataset di /kaggle/input (path mount berubah antara CLI 2.x dan lama).

    - CLI 2.x:  /kaggle/input/datasets/<owner>/<slug>/...
    - Skema lama: /kaggle/input/<slug>/...
    Tidak ada hardcode path: cari dari daftar file ter-mount.
    """
    mounted = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == csv_name:
                mounted.append(os.path.join(root, f))
    if not mounted:
        raise FileNotFoundError(
            f"Dataset '{csv_name}' tidak ditemukan di /kaggle/input. "
            "Cek dataset_sources di kernel-metadata.json."
        )
    return mounted[0]


def load_dataframe(
    csv_name: str = CSV_NAME,
    col_text: str = COL_TEXT,
    col_label: str = COL_LABEL,
) -> pd.DataFrame:
    """Muat CSV dataset dengan validasi kolom teks eksplisit."""
    path = find_dataset_csv(csv_name)
    print("CSV ditemukan di:", path)
    df = pd.read_csv(path)
    if col_text not in df.columns:
        raise ValueError(
            f"Kolom '{col_text}' tidak ditemukan di CSV. Kolom tersedia: {df.columns.tolist()}"
        )
    df[col_text] = df[col_text].fillna("").astype(str)
    print(f"Kolom BERT terpilih: {col_text} | Total baris: {len(df)}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    col_text: str | None = None,
    col_label: str | None = None,
) -> dict[str, np.ndarray]:
    """Split 80:20 (test) lalu 90:10 (val) — protokol konsisten semua eksperimen."""
    ct = col_text or COL_TEXT
    cl = col_label or COL_LABEL
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[cl]
    )
    X_train = train_df[ct].values
    X_test = test_df[ct].values
    y_train = train_df[cl].values
    y_test = test_df[cl].values

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


class EncodedDataset(Dataset):
    """Dataset PyTorch dari encodings pre-tokenized (input_ids + attention_mask + labels)."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: Any):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels.values if isinstance(labels, pd.Series) else np.array(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def build_simulated_scenario(
    texts: Any,
    labels: Any,
    targets: dict[int, int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bentuk skenario ketimpangan kelas deterministik HANYA dari partisi train (zero leakage).

    Replikasi persis protokol experiments/generate_simulated_data.py:
    sample n sampel per kelas tanpa pengembalian (random_state=seed) lalu shuffle penuh.
    """
    df = pd.DataFrame({"text": np.array(texts), "label": np.array(labels)})
    dfs = []
    for label_val, target_n in targets.items():
        class_subset = df[df["label"] == label_val]
        if len(class_subset) < target_n:
            raise ValueError(
                f"Insufficient samples for class {label_val}: available {len(class_subset)}, requested {target_n}"
            )
        dfs.append(class_subset.sample(n=target_n, replace=False, random_state=seed))
    scenario_df = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return scenario_df["text"].values, scenario_df["label"].values.astype(np.int64)


def apply_balancing(
    texts: Any,
    labels: Any,
    strategy: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float] | None]:
    """Terapkan strategi penyeimbangan pada teks latih (mirror protokol M8).

    Returns: (texts_balanced, labels_balanced, class_weight_list_or_None).
    """
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    if strategy == "baseline":
        return texts_arr, labels_arr, None

    if strategy == "class_weight":
        classes = np.array([0, 1, 2])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_arr)
        return texts_arr, labels_arr, [float(w) for w in weights]

    if strategy == "ros":
        unique, counts = np.unique(labels_arr, return_counts=True)
        max_c = max(counts)
        dfs_x, dfs_y = [], []
        rng = np.random.default_rng(seed)
        for cls_val in unique:
            idx = np.where(labels_arr == cls_val)[0]
            if len(idx) < max_c:
                resampled_idx = rng.choice(idx, size=max_c, replace=True)
                dfs_x.append(texts_arr[resampled_idx])
                dfs_y.append(labels_arr[resampled_idx])
            else:
                dfs_x.append(texts_arr[idx])
                dfs_y.append(labels_arr[idx])
        X_bal = np.concatenate(dfs_x, axis=0)
        y_bal = np.concatenate(dfs_y, axis=0)
        perm = rng.permutation(len(y_bal))
        return X_bal[perm], y_bal[perm], None

    if strategy == "rus":
        unique, counts = np.unique(labels_arr, return_counts=True)
        min_c = min(counts)
        dfs_x, dfs_y = [], []
        rng = np.random.default_rng(seed)
        for cls_val in unique:
            idx = np.where(labels_arr == cls_val)[0]
            if len(idx) > min_c:
                resampled_idx = rng.choice(idx, size=min_c, replace=False)
                dfs_x.append(texts_arr[resampled_idx])
                dfs_y.append(labels_arr[resampled_idx])
            else:
                dfs_x.append(texts_arr[idx])
                dfs_y.append(labels_arr[idx])
        X_bal = np.concatenate(dfs_x, axis=0)
        y_bal = np.concatenate(dfs_y, axis=0)
        perm = rng.permutation(len(y_bal))
        return X_bal[perm], y_bal[perm], None

    raise ValueError(f"Unknown strategy: {strategy} (pilihan: baseline, class_weight, ros, rus)")


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
