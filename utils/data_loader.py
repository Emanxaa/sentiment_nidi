"""
Data Loader Module (Milestone M1)
================================
Provides modular and reproducible dataset loading and stratified splitting
for Indonesian flood sentiment classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(
    filepath: str | Path,
    text_col: str = "processed_text_v2",
    label_col: str = "label"
) -> pd.DataFrame:
    """
    Load dataset from CSV and validate required columns.

    Args:
        filepath: Path to dataset CSV.
        text_col: Name of the text column.
        label_col: Name of the label column.

    Returns:
        pd.DataFrame containing valid rows.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing or dataframe is empty.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)
    
    missing_cols = [col for col in (text_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}. Available: {list(df.columns)}"
        )

    # Ensure no nulls in critical columns
    initial_count = len(df)
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)

    if len(df) == 0:
        raise ValueError(f"Dataset is empty after dropping nulls from {path}")

    if len(df) < initial_count:
        print(f"[!] Dropped {initial_count - len(df)} rows with null text or label.")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split into Train+Val and Test sets (e.g. 80:20).

    Args:
        df: Input DataFrame.
        test_size: Proportion for the test split (default: 0.2).
        random_state: Random seed for reproducibility.
        stratify_col: Column name to use for stratified sampling.

    Returns:
        Tuple of (train_val_df, test_df)
    """
    stratify = df[stratify_col] if (stratify_col and stratify_col in df.columns) else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_validation_split(
    train_val_df: pd.DataFrame,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation split strictly from the training set (e.g. 10% of 80% = 8% total).

    Args:
        train_val_df: DataFrame of the combined train+val split.
        val_size: Proportion of train_val_df to allocate to validation (default: 0.1).
        random_state: Random seed for reproducibility.
        stratify_col: Column name to use for stratified sampling.

    Returns:
        Tuple of (train_df, val_df)
    """
    stratify = train_val_df[stratify_col] if (stratify_col and stratify_col in train_val_df.columns) else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_data_splits(
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level helper to execute complete stratified split pipeline using config.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    dataset_cfg = config.get("dataset", {})
    split_cfg = config.get("split", {})
    seed = config.get("seed", 42)

    filepath = dataset_cfg.get("path", "Data/processed/banjir_processed_v2.csv")
    text_col = dataset_cfg.get("text_column", "processed_text_v2")
    label_col = dataset_cfg.get("label_column", "label")

    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.1)
    stratify_col = label_col if split_cfg.get("stratify", True) else None

    # Step 1: Load dataset
    df = load_dataset(filepath, text_col=text_col, label_col=label_col)

    # Step 2: 80:20 Train-Test split
    train_val_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=seed,
        stratify_col=stratify_col
    )

    # Step 3: 90:10 Train-Val split from Train only
    train_df, val_df = create_validation_split(
        train_val_df,
        val_size=val_size,
        random_state=seed,
        stratify_col=stratify_col
    )

    return train_df, val_df, test_df
