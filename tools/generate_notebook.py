"""Generator notebook eksperimen: config YAML -> notebook 10-sel + temp_kernel/<exp>/.

Prinsip:
- Sumber kebenaran ada di src/ - source modul-modulnya DISUNTIK ke sel notebook
  (karena di Kaggle tidak bisa import src/), sehingga notebook self-contained
  dan template sel identik untuk semua eksperimen.
- Setiap eksperimen hanya berbeda file config di configs/<exp_id>.yaml.
- Keluaran: notebooks/<exp_id>.ipynb + temp_kernel/<exp_id>/ (siap kaggle kernels push).

Cara pakai:
    python tools/generate_notebook.py --config configs/exp_p1_ft_sweep.yaml
    python tools/generate_notebook.py --config configs/exp_p2_tapt_mlm.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg_mod

SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
NOTEBOOKS = ROOT / "notebooks"
TEMP_KERNEL = ROOT / "temp_kernel"
OWNER = "emanuelembuaijdak"

# Modul src yang disuntik per family (urutan penting: dependencies dulu)
SRC_MODULES_BY_FAMILY = {
    "hf_lora": ["config.py", "data.py", "model.py", "metrics.py", "trainer_factory.py", "summary.py"],
    "hf_lora_sweep": ["config.py", "data.py", "model.py", "metrics.py", "trainer_factory.py", "summary.py"],
    "hf_lora_v2_suite": ["config.py", "data.py", "model.py", "metrics.py", "trainer_factory.py", "summary.py"],
    "hf_tapt_lora": ["config.py", "data.py", "model.py", "metrics.py", "trainer_factory.py", "summary.py"],
    "keras_lstm": ["config.py", "keras_data.py", "keras_model.py", "metrics.py", "summary.py"],
    "keras_bilstm": ["config.py", "keras_data.py", "keras_model.py", "metrics.py", "summary.py"],
}

PIN_CELL_HF = """\
# P0.3 (replikasi): pin stack era 4.x (transformers 5.0 menurunkan performa).
# torchao 0.10 tidak kompatibel dengan peft -> uninstall dulu.
!pip uninstall -y torchao
!pip install --force-reinstall --no-deps "transformers==4.46.3" "peft==0.13.2" "tokenizers==0.20.3" "huggingface-hub==0.26.5"
"""

PIN_CELL_KERAS = """\
# P0.3 (replikasi): pin stack era 4.x.
!pip uninstall -y torchao
!pip install --force-reinstall --no-deps "transformers==4.46.3" "peft==0.13.2" "tokenizers==0.20.3" "huggingface-hub==0.26.5"
"""

GPU_CELL = """\
# P0.1 (replikasi): paksa 1 GPU (DataParallel menggandakan batch -> undertrained).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

import sys
import torch
import transformers
import peft

assert transformers.__version__.startswith("4.46"), (
    f"transformers {transformers.__version__} bukan pin 4.46 - instalasi bermasalah!"
)
from transformers import TFPreTrainedModel  # bukti tidak ada file campur 5.0

print("python        :", sys.version)
print("torch         :", torch.__version__)
print("transformers  :", transformers.__version__)
print("peft          :", peft.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count     :", torch.cuda.device_count())
"""

SEED_CELL = """\
# =====================================================
# SET SEED
# =====================================================
import random
import numpy as np
import torch
from transformers import set_seed

seed = 42
set_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print("GPU tersedia:", torch.cuda.is_available())
"""

DATASET_CELL = """\
# =====================================================
# DATASET (loader fleksibel + validasi text_bert)
# =====================================================
df = load_dataframe()
split = split_data(df, test_size=0.2, val_size=0.1, random_state=seed)

max_len = CONFIG.get("params", {}).get("max_length", 128) if isinstance(CONFIG.get("params"), dict) else 128
tokenizer = load_tokenizer()
train_dataset = SentimenDataset(split["X_train"], split["y_train"], tokenizer, max_length=max_len)
val_dataset = SentimenDataset(split["X_val"], split["y_val"], tokenizer, max_length=max_len)
test_dataset = SentimenDataset(split["X_test"], split["y_test"], tokenizer, max_length=max_len)
print(f"Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(test_dataset)}")
"""

MODEL_CELL = """\
# =====================================================
# MODEL (build LoRA dari CONFIG)
# =====================================================
p = CONFIG.get("params", {})
model = build_indobertweet_lora(
    dropout=p.get("dropout", 0.3),
    r=p.get("lora_r", 16),
    lora_alpha=p.get("lora_alpha", 32),
)
model.print_trainable_parameters()
"""

GPU_CELL_KERAS = """\
# P0.1 (replikasi): paksa 1 GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

import sys
import tensorflow as tf
print("python :", sys.version)
print("tf     :", tf.__version__)
print("gpus   :", tf.config.list_physical_devices("GPU"))
"""

SEED_CELL_KERAS = """\
# =====================================================
# SET SEED
# =====================================================
import random
import numpy as np
import tensorflow as tf

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
print("GPU tersedia:", tf.config.list_physical_devices("GPU") != [])
"""

DATASET_CELL_KERAS = """\
# =====================================================
# DATASET (loader fleksibel + kolom clean_text_lstm)
# =====================================================
d = load_lstm_data(
    max_words=CONFIG["params"]["max_words"],
    max_len=CONFIG["params"]["max_len"],
    test_size=0.2,
    val_size=0.1,
    random_state=42,
)
"""

MODEL_CELL_KERAS = """\
# =====================================================
# MODEL FACTORY (Keras, dari CONFIG)
# =====================================================
def make_model(variant):
    return build_lstm_model(
        max_words=CONFIG["params"]["max_words"],
        max_len=CONFIG["params"]["max_len"],
        embedding_dim=CONFIG["params"].get("embedding_dim", 128),
        units=variant["units"],
        dropout=variant["dropout"],
        learning_rate=variant["learning_rate"],
        bidirectional=(CONFIG["family"] == "keras_bilstm"),
    )
"""

def _training_keras_cell(config: dict) -> str:
    exp_id = config["exp_id"]
    return f"""\
# =====================================================
# TRAINING (loop variants + sanity collapse)
# =====================================================
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

variants = CONFIG["variants"]
val_results = []
best_model = None
best_f1 = -1
best_variant = None

for v in variants:
    name = v["name"]
    print("\\n" + "=" * 60)
    print("VARIANT:", name, "|", v)
    print("=" * 60)

    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = make_model(v)

    callbacks = [EarlyStopping(
        monitor="val_loss",
        patience=v.get("patience", 5),
        restore_best_weights=v.get("restore_best_weights", True),
    )]
    if v.get("reduce_lr"):
        callbacks.append(ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ))

    class_weight = None
    if v.get("class_weight"):
        classes = np.unique(d["y_train"])
        cw_arr = compute_class_weight("balanced", classes=classes, y=d["y_train"])
        class_weight = dict(enumerate(cw_arr))
        print("Class weights:", class_weight)

    history = model.fit(
        d["X_train_pad"],
        d["y_train"],
        validation_data=(d["X_val_pad"], d["y_val"]),
        epochs=CONFIG["params"]["epochs"],
        batch_size=v.get("batch_size", 32),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    y_val_pred = np.argmax(model.predict(d["X_val_pad"], verbose=0), axis=1)
    acc_val = accuracy_score(d["y_val"], y_val_pred)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        d["y_val"], y_val_pred, average="macro", zero_division=0
    )

    maj = pd.Series(d["y_val"]).mode()[0]
    p_maj = float((d["y_val"] == maj).mean())
    f1_maj = (2 * p_maj / (1 + p_maj)) / 3
    print("Distribusi prediksi val :", pd.Series(y_val_pred).value_counts().sort_index().to_dict())
    print("Baseline mayoritas val  : acc=" + str(round(p_maj, 4)) + " macro_f1=" + str(round(f1_maj, 4)))
    status = "COLLAPSE" if f1_macro <= f1_maj + 1e-6 else "OK"
    print("Val accuracy:", round(acc_val, 4), "| Val macro F1:", round(f1_macro, 4), "| STATUS:", status)

    val_results.append({{
        "name": name,
        "learning_rate": v["learning_rate"],
        "units": v["units"],
        "dropout": v["dropout"],
        "batch_size": v.get("batch_size", 32),
        "class_weight": v.get("class_weight", False),
        "accuracy_val": acc_val,
        "f1_macro_val": f1_macro,
        "status": status,
    }})

    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_model = model
        best_variant = name

df_val = pd.DataFrame(val_results).sort_values("f1_macro_val", ascending=False)
print("\\n=== RINGKASAN VALIDATION ===")
print(df_val.to_string(index=False))
df_val.to_csv("{exp_id}_val.csv", index=False)
print("Best variant (val macro F1):", best_variant, "| F1:", round(best_f1, 4))
"""

def _eval_keras_cell(config: dict) -> str:
    exp_id = config["exp_id"]
    return f"""\
# =====================================================
# EVALUASI TEST (best model) + SIMPAN PROBABILITAS
# =====================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

P = best_model.predict(d["X_test_pad"], verbose=0)
y_pred_test = np.argmax(P, axis=1)

print(classification_report(d["y_test"], y_pred_test, target_names=LABEL_NAMES, zero_division=0))
print("Distribusi prediksi:", pd.Series(y_pred_test).value_counts().sort_index().to_dict())
print("Distribusi aktual  :", pd.Series(d["y_test"]).value_counts().sort_index().to_dict())

hasil = pd.DataFrame({{
    "label_aktual": pd.Series(d["y_test"]),
    "label_prediksi": pd.Series(y_pred_test),
    "prob_negatif": P[:, 0],
    "prob_netral": P[:, 1],
    "prob_positif": P[:, 2],
}})
fname = "{exp_id}_test.csv"
hasil.to_csv(fname, index=False)
print("Tersimpan:", fname)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    d["y_test"], y_pred_test, average="macro", zero_division=0
)
accuracy = accuracy_score(d["y_test"], y_pred_test)

cm = confusion_matrix(d["y_test"], y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.title("Confusion Matrix - {exp_id}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
"""

def _src_cell(family: str) -> str:
    parts = ["from __future__ import annotations",
             "# =====================================================",
             "# SUMBER KEBENARAN: src/ (disuntik oleh tools/generate_notebook.py)",
             "# Jangan edit langsung di notebook - edit src/ lalu generate ulang.",
             "# ====================================================="]
    module_key = family if family in SRC_MODULES_BY_FAMILY else "hf_lora"
    for name in SRC_MODULES_BY_FAMILY[module_key]:
        text = (SRC / name).read_text(encoding="utf-8")
        text = "\n".join(
            ln for ln in text.splitlines()
            if not ln.strip().startswith("from __future__ import")
        )
        if "__main__" in text:
            text = text.split('if __name__ == "__main__":')[0]
        parts.append(f"\n# --- src/{name} ---\n{text}")
    return "\n".join(parts)

def _config_cell(config: dict) -> str:
    return f"# =====================================================\n# CONFIG (dibenamkan dari configs/{config['exp_id']}.yaml)\n# =====================================================\nCONFIG = {cfg_mod.config_repr(config)}\nprint(json.dumps(CONFIG, indent=2))"

def _training_cell(config: dict) -> str:
    p = config["params"]
    loss = config.get("loss", "cross_entropy")
    cw = config.get("class_weight")
    gamma = config.get("gamma", 2.0)
    return f"""\
# =====================================================
# TRAINING (trainer_factory: loss={loss})
# =====================================================
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results_{config['exp_id']}",
    learning_rate={p['learning_rate']!r},
    per_device_train_batch_size={p['batch_size']!r},
    per_device_eval_batch_size={p['batch_size']!r},
    num_train_epochs={p['epochs']!r},
    weight_decay={p.get('weight_decay', 0.01)!r},
    warmup_ratio={p.get('warmup_ratio', 0.0)!r},
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    save_total_limit=1,
)

trainer = build_trainer(
    loss={loss!r},
    class_weight={cw!r},
    gamma={gamma!r},
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
eval_result = trainer.evaluate()
print("Hasil validation:", eval_result)

# --- Sanity check (P0): deteksi collapse ---
preds_val = trainer.predict(val_dataset)
y_val_pred = np.argmax(preds_val.predictions, axis=1)
maj = pd.Series(split["y_val"]).mode()[0]
p_maj = float((split["y_val"] == maj).mean())
f1_maj = (2 * p_maj / (1 + p_maj)) / 3
print("Distribusi prediksi val :", pd.Series(y_val_pred).value_counts().sort_index().to_dict())
print("Baseline mayoritas val  : acc=" + str(round(p_maj, 4)) + " macro_f1=" + str(round(f1_maj, 4)))
status = "COLLAPSE" if eval_result["eval_f1_macro"] <= f1_maj + 1e-6 else "OK"
print("STATUS:", status)
"""

def _training_hf_sweep_cell(config: dict) -> str:
    exp_id = config["exp_id"]
    return f"""\
# =====================================================
# TRAINING SWEEP (loop variants: LR, Epoch, Warmup, Weight Decay, MaxLen)
# =====================================================
from transformers import TrainingArguments

variants = CONFIG["variants"]
val_results = []
best_val_f1 = -1.0
best_variant = None
best_trainer = None
best_test_dataset = test_dataset

for v in variants:
    name = v["name"]
    lr = v["learning_rate"]
    ep = v["epochs"]
    wm = v.get("warmup_ratio", 0.0)
    wd = v.get("weight_decay", 0.0)
    max_len = v.get("max_length", 128)
    bs = v.get("batch_size", 16)

    print("\\n" + "=" * 65)
    print(f"VARIANT: {{name}}")
    print(f"LR: {{lr}} | Epochs: {{ep}} | Warmup: {{wm}} | WD: {{wd}} | MaxLen: {{max_len}} | Batch: {{bs}}")
    print("=" * 65)

    tok = load_tokenizer()
    v_train_ds = SentimenDataset(split["X_train"], split["y_train"], tok, max_length=max_len)
    v_val_ds = SentimenDataset(split["X_val"], split["y_val"], tok, max_length=max_len)
    v_test_ds = SentimenDataset(split["X_test"], split["y_test"], tok, max_length=max_len)

    torch.cuda.empty_cache()
    p = CONFIG.get("params", {{}})
    v_model = build_indobertweet_lora(
        dropout=p.get("dropout", 0.3),
        r=p.get("lora_r", 16),
        lora_alpha=p.get("lora_alpha", 32),
    )

    training_args = TrainingArguments(
        output_dir=f"./results_{{name}}",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=ep,
        warmup_ratio=wm,
        weight_decay=wd,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
    )

    trainer_v = build_trainer(
        loss="cross_entropy",
        model=v_model,
        args=training_args,
        train_dataset=v_train_ds,
        eval_dataset=v_val_ds,
        compute_metrics=compute_metrics,
    )
    trainer_v.train()
    eval_result = trainer_v.evaluate()
    val_f1 = eval_result["eval_f1_macro"]
    val_acc = eval_result["eval_accuracy"]
    print(f"Validation -> Acc: {{val_acc:.4f}} | Macro F1: {{val_f1:.4f}}")

    preds_val = trainer_v.predict(v_val_ds)
    y_val_pred = np.argmax(preds_val.predictions, axis=1)
    maj = pd.Series(split["y_val"]).mode()[0]
    p_maj = float((split["y_val"] == maj).mean())
    f1_maj = (2 * p_maj / (1 + p_maj)) / 3
    status = "COLLAPSE" if val_f1 <= f1_maj + 1e-6 else "OK"
    print("STATUS:", status)

    val_results.append({{
        "name": name,
        "learning_rate": lr,
        "epochs": ep,
        "warmup_ratio": wm,
        "weight_decay": wd,
        "max_length": max_len,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "status": status,
    }})

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_variant = name
        best_trainer = trainer_v
        best_test_dataset = v_test_ds

df_val = pd.DataFrame(val_results).sort_values("val_macro_f1", ascending=False)
print("\\n=== RINGKASAN VALIDATION SWEEP P1 ===")
print(df_val.to_string(index=False))
df_val.to_csv("{exp_id}_val.csv", index=False)
print(f"\\nBest Variant: {{best_variant}} dengan Val Macro F1: {{best_val_f1:.4f}}")

trainer = best_trainer
test_dataset = best_test_dataset
"""

def _tapt_cell(config: dict) -> str:
    p = config["params"]
    return f"""\
# =====================================================
# TAHAP 1: TASK-ADAPTIVE PRETRAINING (MLM Domain Adaptation)
# =====================================================
import math
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer as HFTrainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

print("Memulai Tahap 1: Masked Language Modeling pada Korpus Banjir...")

raw_texts = df[COL_TEXT].dropna().tolist()
print(f"Total tweet korpus untuk TAPT: {{len(raw_texts)}}")

train_texts, val_texts = train_test_split(raw_texts, test_size=0.1, random_state=42)
mlm_train_ds = MLMDataset(train_texts, tokenizer, max_length={p.get('max_length', 128)})
mlm_val_ds = MLMDataset(val_texts, tokenizer, max_length={p.get('max_length', 128)})

mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability={p.get('mlm_probability', 0.15)},
)

mlm_args = TrainingArguments(
    output_dir="./tapt_mlm_checkpoints",
    num_train_epochs={p.get('mlm_epochs', 3)},
    learning_rate={p.get('mlm_learning_rate', 5e-5)},
    per_device_train_batch_size={p.get('mlm_batch_size', 16)},
    per_device_eval_batch_size={p.get('mlm_batch_size', 16)},
    weight_decay={p.get('mlm_weight_decay', 0.01)},
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
)

mlm_trainer = HFTrainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=mlm_train_ds,
    eval_dataset=mlm_val_ds,
    data_collator=data_collator,
)

print("\\n[MLM Training] Menjalankan adaptasi domain TAPT...")
mlm_trainer.train()
eval_mlm = mlm_trainer.evaluate()
perplexity = math.exp(eval_mlm["eval_loss"])
print(f"\\nTAPT Selesai! Eval Loss: {{eval_mlm['eval_loss']:.4f}} | Perplexity: {{perplexity:.4f}}")

TAPT_DIR = "./tapt_domain_checkpoint"
mlm_trainer.save_model(TAPT_DIR)
tokenizer.save_pretrained(TAPT_DIR)
print(f"Checkpoint TAPT tersimpan di: {{TAPT_DIR}}")

# =====================================================
# TAHAP 2: DOWNSTREAM LoRA CLASSIFICATION (dari TAPT Checkpoint)
# =====================================================
print("\\nMemulai Tahap 2: Supervised LoRA Fine-Tuning pada Checkpoint TAPT...")

torch.cuda.empty_cache()
model = build_indobertweet_lora(
    dropout={p.get('dropout', 0.3)},
    r={p.get('lora_r', 16)},
    lora_alpha={p.get('lora_alpha', 32)},
    pretrained_model_name_or_path=TAPT_DIR,
)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results_tapt_lora",
    learning_rate={p.get('ft_learning_rate', 2e-4)},
    per_device_train_batch_size={p.get('ft_batch_size', 16)},
    per_device_eval_batch_size={p.get('ft_batch_size', 16)},
    num_train_epochs={p.get('ft_epochs', 8)},
    warmup_ratio={p.get('ft_warmup_ratio', 0.1)},
    weight_decay={p.get('ft_weight_decay', 0.01)},
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    save_total_limit=1,
)

trainer = build_trainer(
    loss="cross_entropy",
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_result = trainer.evaluate()
print("\\nHasil validation downstream LoRA:", eval_result)
"""

def _eval_cell(config: dict) -> str:
    exp_id = config["exp_id"]
    return f"""\
# =====================================================
# EVALUASI TEST + SIMPAN PROBABILITAS
# =====================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

preds_test = trainer.predict(test_dataset)
logits = preds_test.predictions
y_pred_test = np.argmax(logits, axis=1)
P = softmax_np(logits)

print(classification_report(split["y_test"], y_pred_test, target_names=LABEL_NAMES, zero_division=0))
print("Distribusi prediksi:", pd.Series(y_pred_test).value_counts().sort_index().to_dict())
print("Distribusi aktual  :", pd.Series(split["y_test"]).value_counts().sort_index().to_dict())

hasil = prediction_frame(split["X_test"], split["y_test"], logits)
fname = "{exp_id}_test.csv"
hasil.to_csv(fname, index=False)
print("Tersimpan:", fname)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    split["y_test"], y_pred_test, average="macro", zero_division=0
)
accuracy = accuracy_score(split["y_test"], y_pred_test)

cm = confusion_matrix(split["y_test"], y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.title("Confusion Matrix - {exp_id}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
"""

def _summary_cell(config: dict) -> str:
    exp_id = config["exp_id"]
    return f"""\
# =====================================================
# SAVE ARTIFACT + AUTO EXPERIMENT SUMMARY
# =====================================================
metrics = {{
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
}}
summary = experiment_summary(
    exp_id={exp_id!r},
    config=CONFIG,
    metrics=metrics,
    csv_path=\"{exp_id}_test.csv\",
    out_path=\"{exp_id}_summary.json\",
)
"""

def _dataset_cell_v2_suite(config: dict) -> str:
    return """\
# =====================================================
# DATASET (banjir_processed_v2.csv + processed_text_v2)
# =====================================================
import pandas as pd

CSV_V2 = CONFIG.get("dataset_csv", "banjir_processed_v2.csv")
COL_V2 = CONFIG.get("text_col", "processed_text_v2")
COL_LABEL_V2 = CONFIG.get("label_col", "label")

df = load_dataframe(csv_name=CSV_V2, col_text=COL_V2, col_label=COL_LABEL_V2)
split = split_data(df, test_size=0.2, val_size=0.1, random_state=42, col_text=COL_V2, col_label=COL_LABEL_V2)

max_len = CONFIG.get("params", {}).get("max_length", 128)
tokenizer = load_tokenizer()


def make_encoded_ds(texts, labels):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return EncodedDataset(enc, labels)


train_dataset = make_encoded_ds(split["X_train"], split["y_train"])
val_dataset = make_encoded_ds(split["X_val"], split["y_val"])
test_dataset = make_encoded_ds(split["X_test"], split["y_test"])
print(f"Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(test_dataset)}")
print("Distribusi train:", pd.Series(split['y_train']).value_counts().sort_index().to_dict())
print("Distribusi val  :", pd.Series(split['y_val']).value_counts().sort_index().to_dict())
print("Distribusi test :", pd.Series(split['y_test']).value_counts().sort_index().to_dict())
"""


def _suite_cell(config: dict) -> str:
    return """\
# =====================================================
# SUITE: 4 VARIAN EMPIRIS + 5 SIMULASI x 3 SEEDS (27 RUNS)
# =====================================================
import gc
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

p = CONFIG["params"]
LR = p["learning_rate"]
EPOCHS = p["epochs"]
BS = p["batch_size"]
WARMUP = p["warmup_ratio"]
WD = p["weight_decay"]
PATIENCE = p.get("patience", 2)
SEEDS = CONFIG["seeds"]
EMPIRICAL_VARIANTS = CONFIG["empirical_variants"]
SIMULATIONS = CONFIG["simulations"]
SIM_STRATEGIES = CONFIG["sim_strategies"]

results_rows = []


def run_suite_trial(part, run_id, texts_tr, y_tr, strategy, scenario_id, seed):
    set_seed(seed)
    torch.cuda.empty_cache()

    texts_b, y_b, cw = apply_balancing(texts_tr, y_tr, strategy, seed)
    tr_ds = make_encoded_ds(texts_b, y_b)
    print("\\n" + "=" * 70)
    print(f"RUN: {run_id} | strategy={strategy} | seed={seed}")
    print(f"Train n={len(y_b)} | distribusi: {pd.Series(y_b).value_counts().sort_index().to_dict()}")
    if cw is not None:
        print(f"Class weights: {cw}")
    print("=" * 70)

    model = build_indobertweet_lora(
        dropout=p["dropout"],
        r=p["lora_r"],
        lora_alpha=p["lora_alpha"],
    )

    out_dir = f"./results_{run_id}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=LR,
        per_device_train_batch_size=BS,
        per_device_eval_batch_size=BS,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP,
        weight_decay=WD,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    trainer = build_trainer(
        loss="weighted_ce" if strategy == "class_weight" else "cross_entropy",
        class_weight=cw,
        model=model,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    t0 = time.time()
    trainer.train()
    runtime_sec = round(time.time() - t0, 2)
    eval_result = trainer.evaluate()

    preds_test = trainer.predict(test_dataset)
    logits = preds_test.predictions
    y_pred = np.argmax(logits, axis=1)

    y_true = split["y_test"]
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    _, recall_netral, f1_netral, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="macro", zero_division=0
    )

    maj = pd.Series(split["y_val"]).mode()[0]
    p_maj = float((split["y_val"] == maj).mean())
    f1_maj = (2 * p_maj / (1 + p_maj)) / 3
    val_f1 = eval_result["eval_f1_macro"]
    status = "COLLAPSE" if val_f1 <= f1_maj + 1e-6 else "OK"

    fname = f"pred_{run_id}.csv"
    prediction_frame(split["X_test"], y_true, logits).to_csv(fname, index=False)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    del model, trainer, tr_ds
    torch.cuda.empty_cache()
    gc.collect()

    row = {
        "part": part,
        "scenario": scenario_id,
        "strategy": strategy,
        "seed": seed,
        "train_n": int(len(y_b)),
        "val_accuracy": round(float(eval_result["eval_accuracy"]), 4),
        "val_macro_f1": round(float(val_f1), 4),
        "test_accuracy": round(float(accuracy), 4),
        "test_macro_f1": round(float(f1_macro), 4),
        "test_precision_macro": round(float(precision_macro), 4),
        "test_recall_macro": round(float(recall_macro), 4),
        "recall_netral": round(float(recall_netral), 4),
        "f1_netral": round(float(f1_netral), 4),
        "status": status,
        "runtime_sec": runtime_sec,
    }
    results_rows.append(row)
    print(f"  -> Test Acc={accuracy*100:.2f}% | Macro F1={f1_macro*100:.2f}% | "
          f"Recall Netral={recall_netral*100:.2f}% | {status} | {runtime_sec}s")
    return row


# ========== PART A: VARIAN EMPIRIS ==========
print("\\n" + "#" * 75)
print("# PART A: VARIAN EMPIRIS (Baseline, Class Weight, ROS, RUS)")
print("#" * 75)
for variant in EMPIRICAL_VARIANTS:
    for seed in SEEDS:
        run_id = f"emp_{variant}_s{seed}"
        run_suite_trial("empiris", run_id, split["X_train"], split["y_train"], variant, "empiris", seed)

emp_df = pd.DataFrame([r for r in results_rows if r["part"] == "empiris"])
emp_df.to_csv("exp_indobert_v2_empiris_results.csv", index=False)
print("\\n=== HASIL EMPIRIS (TEST) ===")
print(emp_df.sort_values("test_macro_f1", ascending=False).to_string(index=False))

# ========== PART B: SKENARIO SIMULASI KETIMPANGAN ==========
print("\\n" + "#" * 75)
print("# PART B: SIMULASI KETIMPANGAN (1:1:1, 6:3:1, 8:1:1) [+ROS pada 6:3:1 & 8:1:1]")
print("#" * 75)
for sim in SIMULATIONS:
    sc_id = sim["id"]
    X_sc, y_sc = build_simulated_scenario(split["X_train"], split["y_train"], sim["targets"], seed=42)
    print(f"\\nSkenario {sc_id} ({sim['ratio']}): n={len(y_sc)} | "
          f"distribusi: {pd.Series(y_sc).value_counts().sort_index().to_dict()}")
    for strat in SIM_STRATEGIES:
        for seed in SEEDS:
            run_id = f"sim_{sc_id}_{strat}_s{seed}"
            run_suite_trial("simulasi", run_id, X_sc, y_sc, strat, sc_id, seed)

sim_df = pd.DataFrame([r for r in results_rows if r["part"] == "simulasi"])
sim_df.to_csv("exp_indobert_v2_simulasi_results.csv", index=False)
print("\\n=== HASIL SIMULASI (TEST) ===")
print(sim_df.sort_values(["scenario", "strategy", "test_macro_f1"], ascending=[True, True, False]).to_string(index=False))

# ========== MASTER + AGREGASI MEAN+-STD ==========
master_df = pd.DataFrame(results_rows)
master_df.to_csv("exp_indobert_v2_suite_results.csv", index=False)
print("\\n=== MASTER SUITE RESULTS (27 RUNS) ===")
print(master_df.to_string(index=False))

agg = (
    master_df.groupby(["part", "scenario", "strategy"])
    .agg(
        n_seeds=("seed", "count"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", lambda s: s.std(ddof=1)),
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", lambda s: s.std(ddof=1)),
        recall_netral_mean=("recall_netral", "mean"),
        f1_netral_mean=("f1_netral", "mean"),
    )
    .reset_index()
)
agg.to_csv("exp_indobert_v2_suite_summary.csv", index=False)
print("\\n=== RANGKUMAN MEAN +- STD (3 SEEDS) ===")
print(agg.to_string(index=False))
"""


def _suite_summary_cell(config: dict) -> str:
    return """\
# =====================================================
# SAVE ARTIFACT + AUTO EXPERIMENT SUMMARY
# =====================================================
base_agg = agg[(agg["part"] == "empiris") & (agg["strategy"] == "baseline")]
metrics = {
    "total_runs": int(len(master_df)),
    "empiris_baseline_accuracy_mean": float(base_agg["accuracy_mean"].iloc[0]) if len(base_agg) else None,
    "empiris_baseline_macro_f1_mean": float(base_agg["macro_f1_mean"].iloc[0]) if len(base_agg) else None,
    "empiris_baseline_recall_netral_mean": float(base_agg["recall_netral_mean"].iloc[0]) if len(base_agg) else None,
}
summary = experiment_summary(
    exp_id="exp_indobert_v2",
    config=CONFIG,
    metrics=metrics,
    csv_path="exp_indobert_v2_suite_results.csv",
    out_path="exp_indobert_v2_suite_summary.json",
)
"""


def build_notebook(config: dict) -> nbformat.NotebookNode:
    family = config.get("family", "hf_lora")
    if family in ("keras_lstm", "keras_bilstm"):
        cells = [
            new_code_cell(PIN_CELL_KERAS),
            new_code_cell(GPU_CELL_KERAS),
            new_code_cell(_src_cell(family)),
            new_code_cell(_config_cell(config)),
            new_code_cell(SEED_CELL_KERAS),
            new_code_cell(DATASET_CELL_KERAS),
            new_code_cell(MODEL_CELL_KERAS),
            new_code_cell(_training_keras_cell(config)),
            new_code_cell(_eval_keras_cell(config)),
            new_code_cell(_summary_cell(config)),
        ]
    elif family == "hf_lora_sweep":
        cells = [
            new_code_cell(PIN_CELL_HF),
            new_code_cell(GPU_CELL),
            new_code_cell(_src_cell(family)),
            new_code_cell(_config_cell(config)),
            new_code_cell(SEED_CELL),
            new_code_cell(DATASET_CELL),
            new_code_cell(_training_hf_sweep_cell(config)),
            new_code_cell(_eval_cell(config)),
            new_code_cell(_summary_cell(config)),
        ]
    elif family == "hf_lora_v2_suite":
        cells = [
            new_code_cell(PIN_CELL_HF),
            new_code_cell(GPU_CELL),
            new_code_cell(_src_cell(family)),
            new_code_cell(_config_cell(config)),
            new_code_cell(SEED_CELL),
            new_code_cell(_dataset_cell_v2_suite(config)),
            new_code_cell(_suite_cell(config)),
            new_code_cell(_suite_summary_cell(config)),
        ]
    elif family == "hf_tapt_lora":
        cells = [
            new_code_cell(PIN_CELL_HF),
            new_code_cell(GPU_CELL),
            new_code_cell(_src_cell(family)),
            new_code_cell(_config_cell(config)),
            new_code_cell(SEED_CELL),
            new_code_cell(DATASET_CELL),
            new_code_cell(_tapt_cell(config)),
            new_code_cell(_eval_cell(config)),
            new_code_cell(_summary_cell(config)),
        ]
    else:
        cells = [
            new_code_cell(PIN_CELL_HF),
            new_code_cell(GPU_CELL),
            new_code_cell(_src_cell(family)),
            new_code_cell(_config_cell(config)),
            new_code_cell(SEED_CELL),
            new_code_cell(DATASET_CELL),
            new_code_cell(MODEL_CELL),
            new_code_cell(_training_cell(config)),
            new_code_cell(_eval_cell(config)),
            new_code_cell(_summary_cell(config)),
        ]
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb

def _slugify(title: str) -> str:
    import re
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s.strip("-")

def write_kernel_meta(config: dict, out_dir: Path) -> None:
    title = config.get("title", config["exp_id"])
    slug = _slugify(title)
    meta = {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{config['exp_id']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": config.get(
            "dataset_sources",
            ["emanuelembuaijdak/thesis-indobert-processed-data"],
        ),
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

def generate(config_path: str) -> None:
    config = cfg_mod.load_config(config_path)
    exp_id = config["exp_id"]

    nb = build_notebook(config)
    NOTEBOOKS.mkdir(exist_ok=True)
    nb_path = NOTEBOOKS / f"{exp_id}.ipynb"
    nbformat.write(nb, str(nb_path))
    print(f"[OK] notebook: {nb_path}")

    kernel_dir = TEMP_KERNEL / exp_id
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    shutil.copy(nb_path, kernel_dir / f"{exp_id}.ipynb")
    write_kernel_meta(config, kernel_dir)
    print(f"[OK] temp_kernel: {kernel_dir} (siap kaggle kernels push -p temp_kernel/{exp_id})")

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate notebook eksperimen dari config YAML.")
    ap.add_argument("--config", required=True, help="Path config YAML (configs/<exp_id>.yaml)")
    args = ap.parse_args()
    generate(args.config)

if __name__ == "__main__":
    main()
