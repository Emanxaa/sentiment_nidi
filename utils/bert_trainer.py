"""
IndoBERTweet-LoRA Training Factory Module
========================================
Constructs PEFT LoRA models and configures HuggingFace Trainer
with early stopping, learning rate warmup, and checkpoint restoration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from utils.bert_metrics import compute_metrics


def build_indobertweet_lora(config: dict) -> Tuple[torch.nn.Module, LoraConfig]:
    """
    Initialize pretrained base model and attach PEFT LoRA adapter.
    """
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})

    model_name = model_cfg.get("name", "indolem/indobertweet-base-uncased")
    num_labels = model_cfg.get("num_labels", 3)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        modules_to_save=lora_cfg.get("modules_to_save", ["classifier"])
    )

    model = get_peft_model(base_model, peft_config)
    return model, peft_config


def create_trainer(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    config: dict,
    output_dir: Path,
    seed: int = 42,
    callbacks: list | None = None
) -> Trainer:
    """
    Instantiate configured HuggingFace Trainer instance.
    """
    t_cfg = config.get("training", {})
    r_cfg = config.get("runtime", {})

    use_fp16 = torch.cuda.is_available() and r_cfg.get("mixed_precision", True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=t_cfg.get("epochs", 5),
        per_device_train_batch_size=t_cfg.get("batch_size", 16),
        per_device_eval_batch_size=t_cfg.get("batch_size", 16),
        learning_rate=float(t_cfg.get("learning_rate", 2e-5)),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.1),
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=use_fp16,
        gradient_checkpointing=r_cfg.get("gradient_checkpointing", False),
        report_to="none",
        seed=seed,
    )

    all_callbacks = [
        EarlyStoppingCallback(early_stopping_patience=t_cfg.get("early_stopping_patience", 2))
    ]
    if callbacks:
        all_callbacks.extend(callbacks)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=all_callbacks
    )
    return trainer
