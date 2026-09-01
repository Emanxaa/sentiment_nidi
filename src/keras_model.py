"""Model LSTM/BiLSTM (Keras): builder tunggal untuk family keras_lstm/keras_bilstm.

- LSTM: Embedding -> LSTM(units) -> Dropout -> Dense(64, relu) -> Dense(3, softmax)
- BiLSTM: sama, LSTM dibungkus Bidirectional
Sumber kebenaran: file ini disuntik ke sel notebook oleh generator.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    max_words: int = 10000,
    max_len: int = 50,
    embedding_dim: int = 128,
    units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    bidirectional: bool = False,
) -> Sequential:
    """Bangun model Keras (LSTM atau BiLSTM) untuk klasifikasi 3 kelas."""
    model = Sequential()
    model.add(
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)
    )
    if bidirectional:
        model.add(Bidirectional(LSTM(units)))
    else:
        model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
