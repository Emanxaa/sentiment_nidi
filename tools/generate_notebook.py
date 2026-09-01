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

# Modul src yang disuntik (urutan penting: dependencies dulu)
SRC_MODULES = [
    "config.py",
    "data.py",
    "model.py",
    "metrics.py",
    "trainer_factory.py",
    "summary.py",
]

PIN_CELL = """\
# P0.3 (replikasi): pin stack era 4.x (transformers 5.0 menurunkan performa).
# torchao 0.10 tidak kompatibel dengan peft -> uninstall dulu.
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


def _src_cell() -> str:
    """Gabungkan source modul src/ menjadi satu sel (self-contained di Kaggle).

    - `from __future__` hanya boleh di baris pertama file/sel -> dipindah ke atas.
    - `if __name__ == "__main__"` guard dibuang.
    """
    parts = ["from __future__ import annotations",  # wajib baris pertama
             "# =====================================================",
             "# SUMBER KEBENARAN: src/ (disuntik oleh tools/generate_notebook.py)",
             "# Jangan edit langsung di notebook - edit src/ lalu generate ulang.",
             "# ====================================================="]
    for name in SRC_MODULES:
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
    cells = [
        new_code_cell(PIN_CELL),
        new_code_cell(GPU_CELL),
        new_code_cell(_src_cell()),
        new_code_cell(SEED_CELL),
        new_code_cell(DATASET_CELL),
        new_code_cell(_config_cell(config)),
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


def write_kernel_meta(config: dict, out_dir: Path) -> None:
    meta = {
        "id": f"{OWNER}/{config['exp_id']}",
        "title": config.get("title", config["exp_id"]),
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
    print(f"[OK] temp_kernel: {kernel_dir} (siap `kaggle kernels push -p temp_kernel/{exp_id}`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate notebook eksperimen dari config YAML.")
    ap.add_argument("--config", required=True, help="Path config YAML (configs/<exp_id>.yaml)")
    args = ap.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
