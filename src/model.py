"""Model: builder IndoBERTweet-LoRA (sumber kebenaran tunggal)."""
from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
)

MODEL_NAME = "indolem/indobertweet-base-uncased"
ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}
LABEL2ID = {"negatif": 0, "netral": 1, "positif": 2}


def build_indobertweet_lora(
    dropout: float = 0.3,
    r: int = 16,
    lora_alpha: int = 32,
    num_labels: int = 3,
):
    """Bangun model IndoBERTweet + LoRA (r, alpha, dropout) dengan classifier baru."""
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, lora_config)


def load_tokenizer() -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)
