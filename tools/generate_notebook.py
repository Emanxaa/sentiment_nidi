"""Generator notebook eksperimen: config YAML -> notebook 10-sel + temp_kernel/<exp>/.

Prinsip:
- Sumber kebenaran ada di `src/` — source modul-modulnya DISUNTIK ke sel notebook
  (karena di Kaggle tidak bisa import `src/`), sehingga notebook self-contained
  dan template sel identik untuk semua eksperimen.
- Setiap eksperimen hanya berbeda file config di `configs/<exp_id>.yaml`.
- Keluaran: `notebooks/<exp_id>.ipynb` + `temp_kernel/<exp_id>/` (siap `kaggle kernels push`).

Cara pakai:
    python tools/generate_notebook.py --config configs/exp_p1_weightedce.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

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
    "keras_lstm": ["config.py", "keras_data.py", "keras_model.py", "metrics.py", "summary.py"],
    "keras_bilstm": ["config.py", "keras_data.py", "keras_model.py", "metrics.py", "summary.py"],
}
SRC_MODULES = SRC_MODULES_BY_FAMILY["hf_lora"]  # default (backward compat)

# Environment pin — family Keras butuh tensorflow (image sudah ada), tetap pin transformers
# tidak wajib untuk Keras; sel ini disesuaikan per family di bawah.
PIN_CELL_HF = """\
# P0.3 (replikasi): pin stack era 4.x (transformers 5.0 menurunkan performa).
# torchao 0.10 tidak kompatibel dengan peft -> uninstall dulu.
!pip uninstall -y torchao
!pip install --force-reinstall --no-deps "transformers==4.46.3" "peft==0.13.2" "tokenizers==0.20.3" "huggingface-hub==0.26.5"
"""

PIN_CELL_KERAS = """\
# P0.3 (replikasi): pin stack era 4.x — transformers tidak dipakai family Keras,
# tapi pip pin tetap untuk konsistensi environment jika sel import menyentuh HF.
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

tokenizer = load_tokenizer()
train_dataset = SentimenDataset(split["X_train"], split["y_train"], tokenizer, max_length=CONFIG["max_length"])
val_dataset = SentimenDataset(split["X_val"], split["y_val"], tokenizer, max_length=CONFIG["max_length"])
test_dataset = SentimenDataset(split["X_test"], split["y_test"], tokenizer, max_length=CONFIG["max_length"])
print(f"Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(test_dataset)}")
"""

MODEL_CELL = """\
# =====================================================
# MODEL (build LoRA dari CONFIG)
# =====================================================
model = build_indobertweet_lora(
    dropout=CONFIG["dropout"],
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
)
model.print_trainable_parameters()
"""

# ---------------------------------------------------------------------------
# Sel-sel family Keras (LSTM/BiLSTM)
# ---------------------------------------------------------------------------
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
    """Loop variants (grid/debug) -> latih tiap konfigurasi -> pilih best by val macro F1.

    Setiap variant diset seed ulang; sanity check collapse per variant.
    Hasil val disimpan ke `<exp_id>_val.csv`; model terbaik dipertahankan.
    """
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

    # --- Sanity check (P0): deteksi collapse ---
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

P = best_model.predict(d["X_test_pad"], verbose=0)  # Keras softmax -> proba per kelas
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
    """Gabungkan source modul src/ menjadi satu sel (self-contained di Kaggle).

    - `from __future__` hanya boleh di baris pertama file/sel -> dipindah ke atas.
    - `if __name__ == "__main__"` guard dibuang.
    """
    parts = ["from __future__ import annotations",  # wajib baris pertama
             "# =====================================================",
             "# SUMBER KEBENARAN: src/ (disuntik oleh tools/generate_notebook.py)",
             "# Jangan edit langsung di notebook - edit src/ lalu generate ulang.",
             "# ====================================================="]
    for name in SRC_MODULES_BY_FAMILY[family]:
        text = (SRC / name).read_text(encoding="utf-8")
        # buang baris from __future__ (sudah disediakan di atas)
        text = "\n".join(
            ln for ln in text.splitlines()
            if not ln.strip().startswith("from __future__ import")
        )
        # potong bagian main guard bila ada
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
    commit = "GENERATED_"  # diisi generator di bawah
    return f"""\
# =====================================================
# SAVE ARTIFACT + AUTO EXPERIMENT SUMMARY
# =====================================================
import json

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
    csv_path="{exp_id}_test.csv",
    out_path="{exp_id}_summary.json",
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
    """Slug Kaggle dari judul: huruf kecil, spasi -> dash, buang karakter non-alnum."""
    import re

    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    return re.sub(r"\s+", "-", s)


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
    if slug != config["exp_id"]:
        print(
            f"[INFO] slug kernel '{slug}' != exp_id '{config['exp_id']}' "
            "(slug diambil dari judul, sesuai perilaku Kaggle)."
        )
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
    print(f"[OK] temp_kernel: {kernel_dir} (siap `kaggle kernels push -p temp_kernel/{exp_id}`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate notebook eksperimen dari config YAML.")
    ap.add_argument("--config", required=True, help="Path config YAML (configs/<exp_id>.yaml)")
    args = ap.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
