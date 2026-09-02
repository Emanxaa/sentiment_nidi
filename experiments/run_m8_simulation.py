"""
Milestone M8 — Part B: Simulation Experiments Runner
====================================================
Runs all 5 balancing strategies across 3 simulated class-distribution scenarios
and 3 random seeds (42, 123, 456):
  Total: 3 scenarios * 5 strategies * 3 seeds = 45 training runs.

Outputs saved to:
  Output/simulated/{scenario}/{strategy}/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Safe stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_dataset, split_dataset, create_validation_split
from utils.evaluator import evaluate_all_splits
from utils.metrics import (
    generate_classification_report_df,
    save_classification_report_csv,
    save_history_csv,
    save_metrics_json,
)
from utils.model_lstm import build_lstm_model
from utils.tokenizer import (
    fit_tokenizer,
    get_vocab_size,
    save_tokenizer,
    texts_to_padded_sequences,
)
from utils.trainer import train_lstm_model
from utils.visualization import plot_confusion_matrix, plot_learning_curves


SEEDS = [42, 123, 456]
SCENARIOS = ["scenario_111", "scenario_631", "scenario_811"]
STRATEGIES = ["baseline", "class_weight", "random_oversampling", "random_undersampling", "smote"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def apply_balancing(
    strategy: str,
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    vocab_size: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor | None]:
    """Apply specific balancing strategy to training sequences and labels."""
    if strategy == "baseline":
        return X_train_seq, y_train, None

    elif strategy == "class_weight":
        classes = np.array([0, 1, 2])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return X_train_seq, y_train, weights_tensor

    elif strategy == "random_oversampling":
        unique, counts = np.unique(y_train, return_counts=True)
        max_c = max(counts)
        dfs_x, dfs_y = [], []
        for cls_val in unique:
            idx = np.where(y_train == cls_val)[0]
            if len(idx) < max_c:
                rng = np.random.default_rng(seed)
                resampled_idx = rng.choice(idx, size=max_c, replace=True)
                dfs_x.append(X_train_seq[resampled_idx])
                dfs_y.append(y_train[resampled_idx])
            else:
                dfs_x.append(X_train_seq[idx])
                dfs_y.append(y_train[idx])
        X_bal = np.concatenate(dfs_x, axis=0)
        y_bal = np.concatenate(dfs_y, axis=0)
        # Shuffle
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y_bal))
        return X_bal[perm], y_bal[perm], None

    elif strategy == "random_undersampling":
        unique, counts = np.unique(y_train, return_counts=True)
        min_c = min(counts)
        dfs_x, dfs_y = [], []
        for cls_val in unique:
            idx = np.where(y_train == cls_val)[0]
            if len(idx) > min_c:
                rng = np.random.default_rng(seed)
                resampled_idx = rng.choice(idx, size=min_c, replace=False)
                dfs_x.append(X_train_seq[resampled_idx])
                dfs_y.append(y_train[resampled_idx])
            else:
                dfs_x.append(X_train_seq[idx])
                dfs_y.append(y_train[idx])
        X_bal = np.concatenate(dfs_x, axis=0)
        y_bal = np.concatenate(dfs_y, axis=0)
        # Shuffle
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y_bal))
        return X_bal[perm], y_bal[perm], None

    elif strategy == "smote":
        unique, counts = np.unique(y_train, return_counts=True)
        # If already balanced, return original
        if len(set(counts)) == 1:
            return X_train_seq, y_train, None
        min_count = min(counts)
        k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train_seq, y_train)
        X_res = np.clip(np.round(X_res).astype(np.int64), 0, vocab_size - 1)
        return X_res, y_res, None

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_all_simulations(config_path: str = "configs/lstm_config.yaml") -> dict:
    """Execute all 45 simulation experiments."""
    with open(PROJECT_ROOT / config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set CPU optimization
    if not torch.cuda.is_available():
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("          MILESTONE M8 -- SIMULATION EXPERIMENTS (45 RUNS)")
    print("=" * 75)
    print(f"[*] Device             : {device}")
    print(f"[*] Scenarios (3)      : {SCENARIOS}")
    print(f"[*] Strategies (5)     : {STRATEGIES}")
    print(f"[*] Seeds (3)          : {SEEDS}")
    print(f"[*] Total Experiments  : {len(SCENARIOS) * len(STRATEGIES) * len(SEEDS)}")
    print("=" * 75)

    # Load canonical validation and test splits from raw dataset (seed 42)
    dataset_cfg = config.get("dataset", {})
    raw_path = PROJECT_ROOT / dataset_cfg.get("path", "Data/processed/banjir_processed_v2.csv")
    text_col = dataset_cfg.get("text_column", "processed_text_v2")
    label_col = dataset_cfg.get("label_column", "label")

    raw_df = load_dataset(raw_path, text_col=text_col, label_col=label_col)
    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.1)
    stratify_col = label_col if split_cfg.get("stratify", True) else None

    # Canonical Val & Test sets (fixed seed 42)
    train_val_df, test_df = split_dataset(raw_df, test_size=test_size, random_state=42, stratify_col=stratify_col)
    orig_train_df, val_df = create_validation_split(train_val_df, val_size=val_size, random_state=42, stratify_col=stratify_col)

    # Fit canonical tokenizer on original training data (zero leakage)
    tok_cfg = config.get("tokenizer", {})
    max_words = tok_cfg.get("max_words", 20000)
    oov_token = tok_cfg.get("oov_token", "<OOV>")
    max_length = tok_cfg.get("max_length", 128)
    padding = tok_cfg.get("padding", "post")
    truncating = tok_cfg.get("truncating", "post")

    canonical_tokenizer = fit_tokenizer(orig_train_df[text_col].values, max_words=max_words, oov_token=oov_token)
    vocab_size = get_vocab_size(canonical_tokenizer, max_words=max_words)

    # Canonical Val & Test sequences
    X_val_seq = texts_to_padded_sequences(canonical_tokenizer, val_df[text_col].values, max_length, padding, truncating)
    X_test_seq = texts_to_padded_sequences(canonical_tokenizer, test_df[text_col].values, max_length, padding, truncating)
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values

    batch_size = config.get("training", {}).get("batch_size", 16)
    val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_runs = len(SCENARIOS) * len(STRATEGIES) * len(SEEDS)
    current_run = 0
    all_results = {}
    overall_start = time.time()

    for sc_id in SCENARIOS:
        sc_file = PROJECT_ROOT / "Data" / "simulated" / f"{sc_id}.csv"
        sc_train_df = load_dataset(sc_file, text_col=text_col, label_col=label_col)
        sc_dir = PROJECT_ROOT / "Output" / "simulated" / sc_id
        sc_dir.mkdir(parents=True, exist_ok=True)
        all_results[sc_id] = {}

        # Transform scenario training texts
        X_sc_train_seq = texts_to_padded_sequences(canonical_tokenizer, sc_train_df[text_col].values, max_length, padding, truncating)
        y_sc_train = sc_train_df[label_col].values

        for strat in STRATEGIES:
            strat_dir = sc_dir / strat
            strat_dir.mkdir(parents=True, exist_ok=True)
            strat_seed_results = {}

            for seed in SEEDS:
                current_run += 1
                run_start = time.time()
                set_seed(seed)
                seed_dir = strat_dir / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                print(f"[{current_run:02d}/{total_runs}] Running {sc_id} | {strat.upper()} | Seed {seed}...", end=" ", flush=True)

                # Apply balancing
                X_train_bal, y_train_bal, class_weights_tensor = apply_balancing(
                    strategy=strat,
                    X_train_seq=X_sc_train_seq,
                    y_train=y_sc_train,
                    vocab_size=vocab_size,
                    seed=seed
                )

                train_dataset = TensorDataset(torch.tensor(X_train_bal, dtype=torch.long), torch.tensor(y_train_bal, dtype=torch.long))
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Build & Train Model
                model = build_lstm_model(config, vocab_size=vocab_size)
                model, history_df, t_time = train_lstm_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    output_dir=seed_dir,
                    device=device,
                    class_weights=class_weights_tensor
                )

                # Evaluate
                loaders = {
                    "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
                    "val": val_loader,
                    "test": test_loader,
                }
                eval_results = evaluate_all_splits(model, loaders, device)

                # Per-seed deliverables
                save_history_csv(history_df, seed_dir / "history.csv")
                class_names = ["Negative", "Neutral", "Positive"]
                report_df = generate_classification_report_df(eval_results["test"]["y_true"], eval_results["test"]["y_pred"], class_names)
                save_classification_report_csv(report_df, seed_dir / "classification_report.csv")

                test_metrics = eval_results["test"]["metrics"]
                val_macro_f1 = eval_results["val"]["metrics"]["macro_f1"]
                metrics_payload = {
                    "accuracy": test_metrics["accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                    "validation_macro_f1": val_macro_f1,
                    "training_time": round(t_time, 2),
                    "train_metrics": eval_results["train"]["metrics"],
                    "val_metrics": eval_results["val"]["metrics"],
                    "test_metrics": test_metrics,
                    "scenario": sc_id,
                    "strategy": strat,
                    "seed": seed
                }
                save_metrics_json(metrics_payload, seed_dir / "metrics.json")
                plot_learning_curves(history_df, seed_dir / "loss_curve.png", seed_dir / "accuracy_curve.png")

                for split_key, fname, title_split in [
                    ("train", "confusion_train.png", "Train"),
                    ("val", "confusion_val.png", "Validation"),
                    ("test", "confusion_test.png", "Test"),
                ]:
                    plot_confusion_matrix(
                        y_true=eval_results[split_key]["y_true"],
                        y_pred=eval_results[split_key]["y_pred"],
                        output_path=seed_dir / fname,
                        labels=class_names,
                        title=f"{sc_id} {strat.upper()} Confusion Matrix ({title_split})"
                    )

                strat_seed_results[seed] = {
                    "metrics": metrics_payload,
                    "history": history_df,
                    "report_df": report_df,
                    "eval_results": eval_results
                }

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"Done in {time.time() - run_start:.1f}s | Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['macro_f1']:.4f}")

            # Populate parent folder with best seed representative artifacts
            best_s = max(SEEDS, key=lambda s: strat_seed_results[s]["metrics"]["macro_f1"])
            best_res = strat_seed_results[best_s]

            save_history_csv(best_res["history"], strat_dir / "history.csv")
            save_classification_report_csv(best_res["report_df"], strat_dir / "classification_report.csv")
            plot_learning_curves(best_res["history"], strat_dir / "loss_curve.png", strat_dir / "accuracy_curve.png")
            for split_key, fname, title_split in [
                ("train", "confusion_train.png", "Train"),
                ("val", "confusion_val.png", "Validation"),
                ("test", "confusion_test.png", "Test"),
            ]:
                plot_confusion_matrix(
                    y_true=best_res["eval_results"][split_key]["y_true"],
                    y_pred=best_res["eval_results"][split_key]["y_pred"],
                    output_path=strat_dir / fname,
                    labels=class_names,
                    title=f"{sc_id} {strat.upper()} Confusion Matrix ({title_split})"
                )

            # Strategy summary
            accs = [strat_seed_results[s]["metrics"]["accuracy"] for s in SEEDS]
            f1s = [strat_seed_results[s]["metrics"]["macro_f1"] for s in SEEDS]
            precs = [strat_seed_results[s]["metrics"]["precision"] for s in SEEDS]
            recs = [strat_seed_results[s]["metrics"]["recall"] for s in SEEDS]

            strat_summary = {
                "scenario": sc_id,
                "strategy": strat,
                "best_seed": best_s,
                "accuracy": {"mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs, ddof=1)), 4)},
                "macro_f1": {"mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s, ddof=1)), 4)},
                "precision": {"mean": round(float(np.mean(precs)), 4), "std": round(float(np.std(precs, ddof=1)), 4)},
                "recall": {"mean": round(float(np.mean(recs)), 4), "std": round(float(np.std(recs, ddof=1)), 4)},
                "seeds": {str(s): strat_seed_results[s]["metrics"] for s in SEEDS}
            }
            save_metrics_json(strat_summary, strat_dir / "metrics.json")
            all_results[sc_id][strat] = strat_summary

    total_elapsed = time.time() - overall_start
    print(f"\n[+] All 45 simulation experiments completed in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)!")

    # Save raw simulated summary json
    sim_summary_file = PROJECT_ROOT / "Output" / "simulated" / "simulated_results_raw.json"
    with open(sim_summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    return all_results


if __name__ == "__main__":
    run_all_simulations()
