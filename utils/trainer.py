"""
Trainer Module (Milestone M1)
=============================
Handles PyTorch LSTM model training with:
- Mixed precision training (torch.amp / torch.cuda.amp & GradScaler)
- CuDNN benchmark optimization
- Early stopping & best model checkpointing
- Training history recording
- Automatic restoration of best model and CUDA cache cleanup
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: str | Path,
    device: torch.device | None = None,
    class_weights: torch.Tensor | None = None
) -> Tuple[nn.Module, pd.DataFrame, float]:
    """
    Train LSTM model with mixed precision, early stopping, and checkpointing.

    Args:
        model: SentimentLSTM model instance.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        config: Configuration dictionary.
        output_dir: Directory where best_model.pt should be saved.
        device: PyTorch device (auto-detected if None).
        class_weights: Optional 1D Tensor of class weights for CrossEntropyLoss.

    Returns:
        Tuple of (best_model, history_df, training_time_seconds)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_model.pt"

    training_cfg = config.get("training", {})
    epochs = training_cfg.get("epochs", 20)
    learning_rate = float(training_cfg.get("learning_rate", 0.0005))
    patience = training_cfg.get("early_stopping_patience", 3)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tesla T4 GPU Optimization
    use_cuda = (device.type == "cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    model = model.to(device)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize AMP Scaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"[*] Starting training on device: {device} (Mixed Precision: {use_cuda}, Max Epochs: {epochs}, Patience: {patience})")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ----------------- TRAIN PHASE -----------------
        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda)

            with autocast_ctx:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(batch_y.detach().cpu().numpy())

        train_loss = train_loss_sum / len(train_preds)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro")

        # ----------------- VALIDATION PHASE -----------------
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda)
                else:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda)

                with autocast_ctx:
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                val_loss_sum += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.detach().cpu().numpy())

        val_loss = val_loss_sum / len(val_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "train_macro_f1": float(train_f1),
            "val_macro_f1": float(val_f1),
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # ----------------- CHECKPOINT & EARLY STOPPING -----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [+] Checkpoint saved to {checkpoint_path.name} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] No improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    training_time = time.time() - start_time
    print(f"[*] Training finished in {training_time:.2f} seconds.")

    # ----------------- RESTORE BEST MODEL -----------------
    if checkpoint_path.exists():
        print(f"[*] Restoring best model weights from epoch {best_epoch} ({checkpoint_path.name})...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Free CUDA memory cache
    if use_cuda:
        torch.cuda.empty_cache()
        print("[*] CUDA cache cleared.")

    history_df = pd.DataFrame(history)
    return model, history_df, training_time
