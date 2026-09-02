"""
Tokenizer Module (Milestone M1)
==============================
Provides Keras Tokenizer wrapper for Indonesian text sequences.
Ensures strictly zero data leakage by fitting exclusively on the training split.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def fit_tokenizer(
    train_texts: Sequence[str],
    max_words: int = 20000,
    oov_token: str = "<OOV>"
) -> Tokenizer:
    """
    Fit Keras Tokenizer ONLY on the training split.

    Args:
        train_texts: Sequence of training text strings.
        max_words: Maximum number of most frequent words to keep.
        oov_token: Token to represent out-of-vocabulary words.

    Returns:
        Fitted Keras Tokenizer instance.
    """
    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token=oov_token,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts([str(t) for t in train_texts])
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    max_length: int = 128,
    padding: str = "post",
    truncating: str = "post"
) -> np.ndarray:
    """
    Convert text sequences to post-padded integer arrays.

    Args:
        tokenizer: Fitted Keras Tokenizer.
        texts: Sequence of text strings.
        max_length: Maximum sequence length.
        padding: 'post' or 'pre' padding (default: 'post').
        truncating: 'post' or 'pre' truncation (default: 'post').

    Returns:
        2D numpy array of shape (N, max_length) with dtype int64.
    """
    sequences = tokenizer.texts_to_sequences([str(t) for t in texts])
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding=padding,
        truncating=truncating
    )
    return np.array(padded, dtype=np.int64)


def get_vocab_size(tokenizer: Tokenizer, max_words: int | None = 20000) -> int:
    """
    Calculate effective vocabulary size (including index 0 for padding).

    Args:
        tokenizer: Fitted Tokenizer.
        max_words: Configured maximum words.

    Returns:
        Integer vocabulary size.
    """
    actual_words = len(tokenizer.word_index)
    if max_words is not None and max_words > 0:
        # +1 because index 0 is reserved for padding, and indices range 1..num_words
        return min(actual_words + 1, max_words + 1)
    return actual_words + 1


def save_tokenizer(tokenizer: Tokenizer, filepath: str | Path) -> None:
    """Save fitted tokenizer to disk as pickle."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(filepath: str | Path) -> Tokenizer:
    """Load fitted tokenizer from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {path.resolve()}")
    with open(path, "rb") as f:
        return pickle.load(f)
