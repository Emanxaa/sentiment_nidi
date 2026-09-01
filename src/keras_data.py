"""Data untuk LSTM/BiLSTM (Keras): loader fleksibel + tokenizer + padding.

Sumber kebenaran loading data untuk eksperimen Keras (family keras_lstm /
keras_bilstm). Sel notebook menyuntik source file ini (via generator) sehingga
self-contained di Kaggle. Data diambil dari kolom `clean_text_lstm` di CSV
dataset (label corrected), split 80:20 lalu 90:10 (protokol sama dgn eksperimen HF).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

COL_TEXT = "clean_text_lstm"
COL_LABEL = "label"
CSV_NAME = "data_preprocessed_with_emoticon.csv"


def find_dataset_csv() -> str:
    """Cari CSV dataset di /kaggle/input (path mount CLI 2.x vs lama)."""
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


def load_lstm_data(
    max_words: int = 10000,
    max_len: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load CSV, split stratify 80:20 -> 90:10, tokenizer, pad_sequences.

    Mengembalikan dict: X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test.
    """
    path = find_dataset_csv()
    print("CSV ditemukan di:", path)
    df = pd.read_csv(path)
    if COL_TEXT not in df.columns:
        raise ValueError(
            f"Kolom '{COL_TEXT}' tidak ditemukan di CSV. Kolom tersedia: {df.columns.tolist()}"
        )
    if COL_LABEL not in df.columns:
        raise ValueError(f"Kolom '{COL_LABEL}' tidak ditemukan di CSV.")

    df[COL_TEXT] = df[COL_TEXT].fillna("").astype(str)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[COL_LABEL]
    )
    X_train, y_train = train_df[COL_TEXT].values, train_df[COL_LABEL].values
    X_test, y_test = test_df[COL_TEXT].values, test_df[COL_LABEL].values

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_final)

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_train_final), maxlen=max_len, padding="post"
    )
    X_val_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding="post"
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding="post"
    )

    print(f"Shape train: {X_train_pad.shape} | val: {X_val_pad.shape} | test: {X_test_pad.shape}")
    print("Distribusi train final:", pd.Series(y_train_final).value_counts().sort_index().to_dict())
    print("Distribusi val        :", pd.Series(y_val).value_counts().sort_index().to_dict())
    print("Distribusi test       :", pd.Series(y_test).value_counts().sort_index().to_dict())

    return {
        "X_train_pad": X_train_pad,
        "X_val_pad": X_val_pad,
        "X_test_pad": X_test_pad,
        "y_train": y_train_final,
        "y_val": y_val,
        "y_test": y_test,
    }
