"""
Model Module (Milestone M1)
===========================
Defines the reusable PyTorch LSTM architecture for Indonesian sentiment classification.
Architecture:
Input -> Embedding -> LSTM -> Dropout -> Dense(3) -> Softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    PyTorch LSTM model for 3-class sentiment classification.
    Supports post-padded sequences by extracting the hidden state
    at the last non-padded token position.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        lstm_units: int = 64,
        dropout: float = 0.2,
        num_classes: int = 3,
        padding_idx: int = 0
    ) -> None:
        """
        Initialize SentimentLSTM architecture.

        Args:
            vocab_size: Total vocabulary size (including padding token at index 0).
            embedding_dim: Dimension of word embeddings.
            lstm_units: Number of hidden units in LSTM layer.
            dropout: Dropout probability.
            num_classes: Number of target sentiment classes (default: 3).
            padding_idx: Index reserved for padding (default: 0).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=lstm_units, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Calculate sequence lengths for post-padded tokens (avoiding zero-padding degradation)
        lengths = (x != self.padding_idx).sum(dim=1).clamp(min=1)

        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # lstm_out: (batch_size, seq_len, lstm_units)
        lstm_out, _ = self.lstm(embedded)

        # Extract hidden state at last actual word token for each sequence in the batch
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden = lstm_out[batch_indices, lengths - 1]

        # Apply dropout
        dropped = self.dropout(last_hidden)

        # Dense linear layer -> (batch_size, num_classes)
        logits = self.fc(dropped)

        return logits


def build_lstm_model(
    config: dict,
    vocab_size: int
) -> SentimentLSTM:
    """
    Construct SentimentLSTM instance from configuration dictionary.

    Args:
        config: Configuration dictionary loaded from YAML.
        vocab_size: Effective vocabulary size from tokenizer.

    Returns:
        Instantiated SentimentLSTM model.
    """
    model_cfg = config.get("model", {})
    embedding_dim = model_cfg.get("embedding_dim", 128)
    lstm_units = model_cfg.get("lstm_units", 64)
    dropout = model_cfg.get("dropout", 0.2)

    return SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout=dropout,
        num_classes=3,
        padding_idx=0
    )
