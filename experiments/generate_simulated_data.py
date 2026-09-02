"""
Milestone M8 — Part A: Simulated Dataset Generation
===================================================
Generates three controlled class-distribution scenarios strictly from the
original Train partition (seed 42) without modifying Validation or Test sets:
  - Scenario A (Balanced 1:1:1): 1,000 : 1,000 : 1,000 = 3,000 samples
  - Scenario B (Moderate Imbalance 6:3:1): 3,000 : 1,500 : 500 = 5,000 samples
  - Scenario C (Severe Imbalance 8:1:1): 3,200 : 400 : 400 = 4,000 samples
"""

import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Safe stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import load_dataset, split_dataset, create_validation_split


def generate_simulated_scenarios(
    config_path: str = "configs/lstm_config.yaml",
    random_seed: int = 42
) -> dict:
    """Generate the three simulated datasets and metadata."""
    with open(PROJECT_ROOT / config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    data_path = PROJECT_ROOT / dataset_cfg.get("path", "Data/processed/banjir_processed_v2.csv")
    text_col = dataset_cfg.get("text_column", "processed_text_v2")
    label_col = dataset_cfg.get("label_column", "label")

    df = load_dataset(data_path, text_col=text_col, label_col=label_col)

    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.1)
    stratify_col = label_col if split_cfg.get("stratify", True) else None

    # Obtain canonical Train split (seed 42)
    train_val_df, test_df = split_dataset(df, test_size=test_size, random_state=random_seed, stratify_col=stratify_col)
    train_df, val_df = create_validation_split(train_val_df, val_size=val_size, random_state=random_seed, stratify_col=stratify_col)

    source_counts = train_df[label_col].value_counts().to_dict()
    print("[*] Source Training Split Class Counts (N=6,226):")
    print(f"    - Negative (0): {source_counts.get(0, 0):,}")
    print(f"    - Neutral  (1): {source_counts.get(1, 0):,}")
    print(f"    - Positive (2): {source_counts.get(2, 0):,}")

    sim_dir = PROJECT_ROOT / "Data" / "simulated"
    sim_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "scenario_111": {
            "name": "Scenario A: Balanced (1:1:1)",
            "ratio": "1:1:1",
            "targets": {0: 1000, 1: 1000, 2: 1000},
            "file": "scenario_111.csv",
        },
        "scenario_631": {
            "name": "Scenario B: Moderately Imbalanced (6:3:1)",
            "ratio": "6:3:1",
            "targets": {0: 3000, 2: 1500, 1: 500},
            "file": "scenario_631.csv",
        },
        "scenario_811": {
            "name": "Scenario C: Highly Imbalanced (8:1:1)",
            "ratio": "8:1:1",
            "targets": {0: 3200, 1: 400, 2: 400},
            "file": "scenario_811.csv",
        }
    }

    metadata = {
        "source_dataset": str(data_path),
        "source_train_counts": {
            "negative": int(source_counts.get(0, 0)),
            "neutral": int(source_counts.get(1, 0)),
            "positive": int(source_counts.get(2, 0)),
            "total": int(len(train_df))
        },
        "validation_counts": {
            "total": int(len(val_df)),
            "unchanged": True
        },
        "test_counts": {
            "total": int(len(test_df)),
            "unchanged": True
        },
        "random_seed": random_seed,
        "scenarios": {}
    }

    for key, sc in scenarios.items():
        dfs = []
        for label_val, target_n in sc["targets"].items():
            class_subset = train_df[train_df[label_col] == label_val]
            if len(class_subset) < target_n:
                raise ValueError(f"Insufficient samples for class {label_val}: available {len(class_subset)}, requested {target_n}")
            sample_df = class_subset.sample(n=target_n, replace=False, random_state=random_seed)
            dfs.append(sample_df)

        scenario_df = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        out_csv = sim_dir / sc["file"]
        scenario_df.to_csv(out_csv, index=False)

        counts_res = scenario_df[label_col].value_counts().to_dict()
        total_res = len(scenario_df)
        print(f"\n[+] Created {sc['name']} -> {out_csv.resolve()}")
        print(f"    Total: {total_res:,} | Neg(0): {counts_res.get(0, 0):,} ({counts_res.get(0, 0)/total_res*100:.1f}%) | Neu(1): {counts_res.get(1, 0):,} ({counts_res.get(1, 0)/total_res*100:.1f}%) | Pos(2): {counts_res.get(2, 0):,} ({counts_res.get(2, 0)/total_res*100:.1f}%)")

        metadata["scenarios"][key] = {
            "name": sc["name"],
            "ratio": sc["ratio"],
            "file": sc["file"],
            "counts": {
                "negative": int(counts_res.get(0, 0)),
                "neutral": int(counts_res.get(1, 0)),
                "positive": int(counts_res.get(2, 0)),
                "total": int(total_res)
            },
            "percentages": {
                "negative": round(counts_res.get(0, 0) / total_res * 100, 2),
                "neutral": round(counts_res.get(1, 0) / total_res * 100, 2),
                "positive": round(counts_res.get(2, 0) / total_res * 100, 2),
            }
        }

    meta_file = sim_dir / "simulation_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n[+] Saved simulation metadata to: {meta_file.resolve()}")

    return metadata


if __name__ == "__main__":
    generate_simulated_scenarios()
