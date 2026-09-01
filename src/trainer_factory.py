"""Trainer Factory: satu fungsi untuk membangun Trainer dengan loss yang dipilih.

Sumber kebenaran tunggal untuk:
- "cross_entropy" -> Trainer standar HF
- "weighted_ce"   -> WeightedTrainer (CrossEntropyLoss(weight=class_weight))
- "focal"         -> FocalLossTrainer (gamma, alpha=class_weight)

Sel notebook menyuntik source file ini, sehingga tidak ada duplikasi kode Trainer
di antar-notebook (akar bug 'trainer_best' & compute_loss ganda di masa lalu).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    """CrossEntropyLoss dengan bobot kelas."""

    def __init__(self, class_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = (
            torch.tensor(class_weight, dtype=torch.float) if class_weight is not None else None
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weight is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weight.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class FocalLossTrainer(Trainer):
    """Focal Loss: FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t), dengan alpha opsional."""

    def __init__(self, gamma=2.0, class_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.class_weight = (
            torch.tensor(class_weight, dtype=torch.float) if class_weight is not None else None
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal = -(1.0 - pt) ** self.gamma * torch.log(pt.clamp(min=1e-8))
        if self.class_weight is not None:
            alpha_t = self.class_weight.to(logits.device).gather(0, labels)
            focal = focal * alpha_t
        loss = focal.mean()
        return (loss, outputs) if return_outputs else loss


def build_trainer(
    loss: str = "cross_entropy",
    class_weight=None,
    gamma: float = 2.0,
    **trainer_kwargs,
) -> Trainer:
    """Bangun Trainer sesuai strategi loss.

    Contoh:
        build_trainer(loss="weighted_ce", class_weight=[0.75, 1.32, 1.03], args=args, ...)
        build_trainer(loss="focal", gamma=2.0, class_weight=[...], args=args, ...)
    """
    if loss == "cross_entropy":
        trainer_cls = Trainer
    elif loss == "weighted_ce":
        trainer_cls = WeightedTrainer
    elif loss == "focal":
        trainer_cls = FocalLossTrainer
    else:
        raise ValueError(f"Loss tidak dikenal: {loss} (pilihan: cross_entropy, weighted_ce, focal)")

    if loss == "weighted_ce":
        return WeightedTrainer(class_weight=class_weight, **trainer_kwargs)
    if loss == "focal":
        return FocalLossTrainer(gamma=gamma, class_weight=class_weight, **trainer_kwargs)
    return Trainer(**trainer_kwargs)
