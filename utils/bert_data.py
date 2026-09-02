"""
IndoBERTweet Data Loading & Tokenization Module
==============================================
Provides PyTorch Dataset abstractions and zero-leakage tokenization
routines for indolem/indobertweet-base-uncased.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from utils.data_loader import load_dataset, split_dataset, create_validation_split


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapper for sequence classification token tensors."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_bert_data(
    config: dict,
    seed: int = 42,
    root_dir: Path | None = None
) -> Tuple[SentimentDataset, SentimentDataset, SentimentDataset, AutoTokenizer, pd.DataFrame]:
    """
    Execute stratified Train-Val-Test split and tokenization with zero leakage.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, tokenizer, raw_df)
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent

    dataset_cfg = config.get("dataset", {})
    data_path = root_dir / dataset_cfg.get("path", "Data/processed/banjir_processed_v2.csv")
    text_col = dataset_cfg.get("text_column", "processed_text_v2")
    label_col = dataset_cfg.get("label_column", "label")

    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.1)

    df = load_dataset(data_path, text_col=text_col, label_col=label_col)

    train_val_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=seed,
        stratify_col=label_col
    )
    train_df, val_df = create_validation_split(
        train_val_df,
        val_size=val_size,
        random_state=seed,
        stratify_col=label_col
    )

    model_name = config.get("model", {}).get("name", "indolem/indobertweet-base-uncased")
    max_length = config.get("model", {}).get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(list(train_df[text_col]), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(list(val_df[text_col]), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    test_enc = tokenizer(list(test_df[text_col]), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    train_dataset = SentimentDataset(train_enc, train_df[label_col].tolist())
    val_dataset = SentimentDataset(val_enc, val_df[label_col].tolist())
    test_dataset = SentimentDataset(test_enc, test_df[label_col].tolist())

    return train_dataset, val_dataset, test_dataset, tokenizer, df
