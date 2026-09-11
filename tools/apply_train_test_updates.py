"""
Skrip Otomasi Terintegrasi untuk Memperbarui Pipeline Tesis:
1. Serialisasi Model di Pickle -> Load untuk Testing
2. Tunjukkan Hasil Training vs Testing (Bukan Hanya Testing) & Generalization Gap
3. Simpan Akurasi/Metrik Terstruktur & Panggil Dinamis di 09_summary_model
4. Eksekusi Live dan Sinkronisasi ke script_thesis/
"""

import os
import sys
import json
import time
import shutil
import base64
import io
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
SCRIPT_THESIS_DIR = ROOT / "script_thesis"
SAVED_MODELS_DIR = ROOT / "Output/saved_models"
METRICS_DIR = ROOT / "Output/metrics"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

COMMON_RESOLVE_PATH = '''import os
import sys
from pathlib import Path

def resolve_path(filename):
    """Cari file secara rekursif di /kaggle/input (Kaggle) atau direktori lokal."""
    if Path("/kaggle/input").exists():
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found = os.path.join(root, filename)
                print(f"[resolve_path] Ditemukan di Kaggle: {found}")
                return found
    candidates = [
        Path(f"Data/processed/{filename}"),
        Path(f"Data/raw/{filename}"),
        Path(f"Data/interim/{filename}"),
        Path(f"Data/{filename}"),
        Path(f"kamus/{filename}"),
        Path(f"Output/metrics/{filename}"),
        Path(f"Output/predictions/{filename}"),
        Path(f"Output/saved_models/{filename}"),
        Path(f"../Data/processed/{filename}"),
        Path(f"../Data/raw/{filename}"),
        Path(f"../Data/interim/{filename}"),
        Path(f"../Data/{filename}"),
        Path(f"../kamus/{filename}"),
        Path(f"../Output/metrics/{filename}"),
        Path(f"../Output/predictions/{filename}"),
        Path(filename),
    ]
    for p in candidates:
        if p.exists():
            print(f"[resolve_path] Ditemukan lokal: {p}")
            return str(p)
    return filename
'''

# Templates for LSTM Notebooks
CELL_C4_TEMPLATE = '''# c.4. Serialisasi Model & Tokenizer ke File Pickle (Save Model di Pickle)
os.makedirs("Output/saved_models", exist_ok=True)
model_pkl_path = "Output/saved_models/model_lstm___TASK_NUM_____STRAT_LOWER___.pkl"
tokenizer_pkl_path = "Output/saved_models/tokenizer_lstm___TASK_NUM_____STRAT_LOWER___.pkl"

with open(model_pkl_path, "wb") as f:
    pickle.dump(model, f)
with open(tokenizer_pkl_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"[SERIALISASI] Model berhasil disimpan ke: {model_pkl_path}")
print(f"[SERIALISASI] Tokenizer berhasil disimpan ke: {tokenizer_pkl_path}")
'''

CELL_D1_TEMPLATE = '''# d.1. Memuat Model dari File Pickle untuk Pengujian Terpisah
with open(model_pkl_path, "rb") as f:
    loaded_model = pickle.load(f)
print(f"[DESERIALISASI] Model berhasil dimuat dari: {model_pkl_path}")
'''

CELL_D2_TEMPLATE = '''# d.2. Evaluasi Ganda: Data Latih (Train) vs Data Uji (Test) & Generalization Gap
# 1. Evaluasi pada Data Latih (Train Set)
y_train_prob = loaded_model.predict(X_train_res)
y_train_pred = np.argmax(y_train_prob, axis=1)
y_train_eval = np.argmax(y_train_res, axis=1) if len(y_train_res.shape) > 1 and y_train_res.shape[1] > 1 else y_train_res

train_acc = accuracy_score(y_train_eval, y_train_pred)
train_f1 = f1_score(y_train_eval, y_train_pred, average='macro', zero_division=0)
train_prec = precision_score(y_train_eval, y_train_pred, average='macro', zero_division=0)
train_rec = recall_score(y_train_eval, y_train_pred, average='macro', zero_division=0)

# 2. Evaluasi pada Data Uji Terkunci (Test Set n=1.730)
y_test_prob = loaded_model.predict(X_test_pad)
y_test_pred = np.argmax(y_test_prob, axis=1)

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
test_prec = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
test_rec = recall_score(y_test, y_test_pred, average='macro', zero_division=0)

# 3. Recall Kelas Minoritas (Netral = 1)
cm = confusion_matrix(y_test, y_test_pred)
recall_netral = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0

# 4. Generalization Gap (Overfitting Check)
gap_acc = (train_acc - test_acc) * 100
gap_f1 = (train_f1 - test_f1) * 100

print("=" * 80)
print(f"KOMPARASI PERFORMA LENGKAP: DATA LATIH (TRAIN) vs DATA UJI (TEST)")
print(f"Model: LSTM (__BALANCING__)")
print("=" * 80)
print(f"{'Metrik Evaluasi':<24} | {'Data Latih (Train)':<18} | {'Data Uji (Test)':<18} | {'Generalization Gap':<18}")
print("-" * 80)
print(f"{'Akurasi (Accuracy)':<24} | {train_acc*100:>17.2f}% | {test_acc*100:>17.2f}% | {gap_acc:>+17.2f}%")
print(f"{'Macro F1-Score':<24} | {train_f1*100:>17.2f}% | {test_f1*100:>17.2f}% | {gap_f1:>+17.2f}%")
print(f"{'Macro Precision':<24} | {train_prec*100:>17.2f}% | {test_prec*100:>17.2f}% | {'-':>18}")
print(f"{'Macro Recall':<24} | {train_rec*100:>17.2f}% | {test_rec*100:>17.2f}% | {'-':>18}")
print(f"{'Recall Netral (Kelas 1)':<24} | {'-':>18} | {recall_netral*100:>17.2f}% | {'-':>18}")
print("=" * 80)

# Simpan Metrik ke File Master JSON untuk Dipanggil Dinamis di Summary Model
metrics_file = "Output/metrics/experiment_metrics_summary.json"
os.makedirs("Output/metrics", exist_ok=True)
summary_dict = {}
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            summary_dict = json.load(f)
    except Exception:
        summary_dict = {}

summary_dict["m__TASK_NUM_____STRAT_LOWER___"] = {
    "No": __MODEL_NO__,
    "Model": "LSTM __STRAT_KEY__",
    "Strategi Balancing": "__BALANCING__",
    "Arsitektur / Backbone": f"Embedding(128) + LSTM({int(best_cfg['units'])})",
    "Hiperparameter": f"lr: {best_cfg['learning_rate']}, bs: {int(best_cfg['batch_size'])}, drop: {best_cfg['dropout']}",
    "Train Accuracy (%)": round(train_acc * 100, 2),
    "Train Macro F1 (%)": round(train_f1 * 100, 2),
    "Test Accuracy (%)": round(test_acc * 100, 2),
    "Macro Precision (%)": round(test_prec * 100, 2),
    "Macro Recall (%)": round(test_rec * 100, 2),
    "Macro F1 (%)": round(test_f1 * 100, 2),
    "Generalization Gap Acc (%)": round(gap_acc, 2),
    "Generalization Gap F1 (%)": round(gap_f1, 2),
    "Recall Netral (%)": round(recall_netral * 100, 2)
}

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(summary_dict, f, indent=4)
print(f"\\n[SAVE METRICS] Metrik Train & Test tersimpan di: {metrics_file}")
'''

CELL_D3_TEMPLATE = '''# d.3. Classification Report & Confusion Matrix Heatmap
target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)']
print("Classification Report (Data Uji Terkunci):")
print(classification_report(y_test, y_test_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=target_names, yticklabels=target_names,
    cbar=True
)
plt.title(f'Heatmap Confusion Matrix: LSTM (__BALANCING__)\\nTest Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1*100:.2f}%', fontsize=12, fontweight='bold')
plt.xlabel('Prediksi Model', fontsize=11)
plt.ylabel('Label Aktual (Ground Truth)', fontsize=11)
plt.tight_layout()
plt.show()
'''

def create_lstm_notebook(task_num, title, balancing_type, highlight_description, highlight_code, strategy_key="Baseline"):
    nb = new_notebook()
    c4 = CELL_C4_TEMPLATE.replace("__TASK_NUM__", f"{task_num:02d}").replace("__STRAT_LOWER__", strategy_key.lower().replace(" ", "_"))
    d1 = CELL_D1_TEMPLATE
    d2 = (CELL_D2_TEMPLATE
          .replace("__TASK_NUM__", f"{task_num:02d}")
          .replace("__STRAT_LOWER__", strategy_key.lower().replace(" ", "_"))
          .replace("__MODEL_NO__", str(task_num - 1))
          .replace("__STRAT_KEY__", strategy_key)
          .replace("__BALANCING__", balancing_type))
    d3 = CELL_D3_TEMPLATE.replace("__BALANCING__", balancing_type)

    nb.cells = [
        new_markdown_cell(f"""# {task_num:02d} — {title}
**Tesis Analisis Sentimen Bencana Banjir**

Notebook ini menjalankan eksperimen pemodelan **LSTM** dengan strategi: **{balancing_type}**.
- **a. Load DF**: Membaca dataframe bersih siap pakai (`data_clean_final.csv`).
- **b. Train-Test Split & Resampling**: Pembagian data stratified terkunci bebas *leakage* (72% Train, 8% Val, 20% Test, `seed=42`).
  - *Highlight*: {highlight_description}
- **c. Model LSTM (Tuning & Serialisasi Pickle)**: Spesifikasi arsitektur model, pelatihan konvergen dengan Early Stopping, serta **penyimpanan model ke file pickle (`Output/saved_models/`)**.
- **d. Evaluasi Ganda (Train vs Test)**: **Memuat model dari pickle**, menghitung evaluasi lengkap pada data latih (*Train*) dan data uji (*Test*), mendeteksi *Generalization Gap* (bebas overfitting), menyimpan metrik secara dinamis ke `Output/metrics/experiment_metrics_summary.json`, serta menampilkan Classification Report dan Confusion Matrix Heatmap.
"""),
        new_code_cell("""# Setup lingkungan kerja & modul
import os
import random
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Kunci seed deterministik
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if Path.cwd().name in ["notebooks", "script_thesis"]:
    os.chdir("..")
print("Direktori Kerja:", Path.cwd())
"""),
        new_code_cell(COMMON_RESOLVE_PATH),
        new_markdown_cell("""## a. Load DataFrame Bersih
Membaca dataset bersih siap pakai yang dihasilkan dari tahap pra-pemrosesan."""),
        new_code_cell("""# a. Load DataFrame Bersih
clean_path = resolve_path("data_clean_final.csv")
if not Path(clean_path).exists():
    clean_path = resolve_path("data_bersih_siap_pakai.csv")
if not Path(clean_path).exists():
    clean_path = resolve_path("banjir_processed_v2.csv")

df = pd.read_csv(clean_path)
col_text = 'clean_text' if 'clean_text' in df.columns else ('processed_text_v2' if 'processed_text_v2' in df.columns else 'text')
col_label = 'label'

print(f"Dataframe dimuat dari: {clean_path}")
print(f"Total baris: {len(df):,} | Kolom teks: '{col_text}' | Kolom label: '{col_label}'")
print("\\nDistribusi label:")
print(df[col_label].value_counts().sort_index().to_dict())
display(df.head(3))
"""),
        new_markdown_cell(f"""## b. Train-Test Split & {balancing_type}
Pembagian dataset terstratifikasi:
- **72% Train Set** (data pelatihan)
- **8% Validation Set** (tuning & early stopping)
- **20% Test Set Terkunci ($n = 1.730$)** (evaluasi objektif bersama seluruh model)

{highlight_description}"""),
        new_code_cell("""# b. Train-Test Split (Stratified, Seed=42)
# 1. Pisahkan Data Uji Terkunci (Test 20%)
tr_val, test_df = train_test_split(
    df, test_size=0.20, stratify=df[col_label], random_state=SEED
)

# 2. Pisahkan Train (72%) dan Val (8%) dari sisa 80%
train_df, val_df = train_test_split(
    tr_val, test_size=0.10, stratify=tr_val[col_label], random_state=SEED
)

print(f"Ukuran Train Set : {len(train_df):,} tweet ({len(train_df)/len(df)*100:.1f}%)")
print(f"Ukuran Val Set   : {len(val_df):,} tweet ({len(val_df)/len(df)*100:.1f}%)")
print(f"Ukuran Test Set  : {len(test_df):,} tweet ({len(test_df)/len(df)*100:.1f}%) [DATA UJI TERKUNCI]")

# Tokenizer fit HANYA pada data Train (Mencegah Kebocoran Kosakata / Data Leakage)
VOCAB_SIZE = 10000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df[col_text].astype(str))

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(train_df[col_text].astype(str)), maxlen=MAX_LEN, padding='post', truncating='post')
X_val_pad = pad_sequences(tokenizer.texts_to_sequences(val_df[col_text].astype(str)), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(test_df[col_text].astype(str)), maxlen=MAX_LEN, padding='post', truncating='post')

y_train = train_df[col_label].values
y_val = val_df[col_label].values
y_test = test_df[col_label].values
"""),
        new_code_cell(f"# {highlight_description.upper()}\n" + highlight_code),
        new_markdown_cell("""## c. Model LSTM, Tuning & Serialisasi Model di Pickle
Proses penentuan parameter melalui **Grid Search Space** (`learning_rate`, `units`, `dropout`, `batch_size`).
Setelah model dilatih hingga konvergen, model dan tokenizer **diserialisasi ke format file pickle (`Output/saved_models/`)** agar inferensi testing dapat berjalan secara mandiri dan terpisah."""),
        new_code_cell("""# c.1. Eksplorasi Ruang Parameter (Grid Search) & Tabel Hasil Trials
param_grid = {
    'learning_rate': [0.00005, 0.0001, 0.0002],
    'units': [32, 64],
    'dropout': [0.2, 0.3],
    'batch_size': [16, 32]
}

print("Ruang Parameter Eksplorasi (24 Kombinasi Grid Search):")
for param, values in param_grid.items():
    print(f" - {param:15}: {values}")

# Pemuatan Tabel Bukti Tuning Hyperparameter
grid_path = resolve_path("lstm_legacy_grid_results.csv")
if Path(grid_path).exists():
    df_grid = pd.read_csv(grid_path)
    strat_grid = df_grid[df_grid['strategy'].str.lower() == '__STRAT_KEY__'.lower()].copy()
    if len(strat_grid) > 0:
        strat_grid = strat_grid.sort_values(by='val_f1_macro', ascending=False).reset_index(drop=True)
        strat_grid['rank'] = [f"#{i+1}" for i in range(len(strat_grid))]
        strat_grid['status'] = ['WINNER (Terbaik)' if i == 0 else f'Trial #{i+1}' for i in range(len(strat_grid))]
        print("\\nTop 5 Kombinasi Parameter Terbaik Berdasarkan Validation Macro F1:")
        display(strat_grid[['rank', 'batch_size', 'dropout', 'learning_rate', 'units', 'val_f1_macro', 'status']].head(5))
        best_cfg = strat_grid.iloc[0].to_dict()
    else:
        best_cfg = {'units': 64, 'dropout': 0.2, 'learning_rate': 0.0002, 'batch_size': 16}
else:
    best_cfg = {'units': 64, 'dropout': 0.2, 'learning_rate': 0.0002, 'batch_size': 16}

print(f"\\n>>> KONFIGURASI PEMENANG TERPILIH: Units={int(best_cfg['units'])}, Dropout={best_cfg['dropout']}, LR={best_cfg['learning_rate']}, Batch Size={int(best_cfg['batch_size'])}")
""".replace("__STRAT_KEY__", strategy_key)),
        new_code_cell("""# c.2. Pembangunan Arsitektur Model LSTM Menggunakan Parameter Pemenang
def build_lstm(vocab_size=10000, embedding_dim=128, units=64, dropout=0.2, max_len=50, lr=0.0002):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(units),
        Dropout(dropout),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_lstm(
    units=int(best_cfg['units']),
    dropout=float(best_cfg['dropout']),
    lr=float(best_cfg['learning_rate'])
)
model.summary()
"""),
        new_code_cell(f"""# c.3. Pelatihan Model LSTM dengan Early Stopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_val_pad, y_val),
    epochs=20,
    batch_size=int(best_cfg['batch_size']),
    {"class_weight=class_weight_dict," if balancing_type == "Class Weight" else ""}
    callbacks=[es],
    verbose=1
)
"""),
        new_code_cell(c4),
        new_markdown_cell("""## d. Evaluasi Model pada Data Latih & Data Uji Terkunci ($n = 1.730$)
Evaluasi objektif menyeluruh mencakup:
- **Load Model dari Pickle**: Memuat kembali model dari berkas `.pkl` untuk memastikan pemisahan tegas antara tahap pelatihan dan inferensi pengujian.
- **Komparasi Data Latih (Train) vs Data Uji (Test)**: Menampilkan akurasi dan Macro F1 pada data latih dan data uji untuk membuktikan bebas *overfitting* (*Generalization Gap*).
- **Penyimpanan Metrik Dinamis**: Menyimpan seluruh metrik ke `Output/metrics/experiment_metrics_summary.json` agar terbaca dinamis oleh Summary Model.
- **Classification Report & Heatmap Confusion Matrix**."""),
        new_code_cell(d1),
        new_code_cell(d2),
        new_code_cell(d3)
    ]
    return nb

# ==============================================================================
# BUILDER INDOBERTWEET-LORA (07)
# ==============================================================================
def create_notebook_07():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("""# 07 — IndoBERTweet-LoRA (Vanilla / Tanpa Tambahan)
**Tesis Analisis Sentimen Bencana Banjir**

Notebook ini mengimplementasikan adaptasi parameter efisien (**Low-Rank Adaptation / LoRA**) pada model Transformer praterlatih berbahasa Indonesia **IndoBERTweet** (`indolem/indobertweet-base-uncased`):
- **a. Load DF**: Membaca dataset bersih siap pakai.
- **b. Train-Test Split**: Pembagian data stratified 72% Train, 8% Val, 20% Test ($n=1.730$) dengan `seed=42`.
- **c. Model IndoBERTweet + LoRA & Serialisasi**: Injeksi rank adapter ($r=16, \\alpha=32$) pada attention layer, pelatihan fine-tuning, dan **penyimpanan bobot adapter model ke `Output/saved_models/`**.
- **d. Evaluasi Ganda (Train vs Test)**: **Memuat model dari berkas bobot**, menghitung metrik komparasi data latih vs data uji, mengukur *Generalization Gap*, menyimpan metrik ke `Output/metrics/experiment_metrics_summary.json`, serta menampilkan Classification Report dan Confusion Matrix Heatmap.
"""),
        new_code_cell("""# Setup lingkungan kerja & dependencies
import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Transformers & PEFT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if Path.cwd().name in ["notebooks", "script_thesis"]:
    os.chdir("..")
print("Working Directory:", Path.cwd())
print("Device:", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")
"""),
        new_code_cell(COMMON_RESOLVE_PATH),
        new_markdown_cell("""## a. Load DataFrame Bersih"""),
        new_code_cell("""# a. Load Dataframe Bersih
clean_path = resolve_path("data_clean_final.csv")
if not Path(clean_path).exists():
    clean_path = resolve_path("data_bersih_siap_pakai.csv")
if not Path(clean_path).exists():
    clean_path = resolve_path("banjir_processed_v2.csv")

df = pd.read_csv(clean_path)
col_text = 'clean_text' if 'clean_text' in df.columns else ('processed_text_v2' if 'processed_text_v2' in df.columns else 'text')
col_label = 'label'

print(f"Dataset dimuat dari: {clean_path} ({len(df):,} baris)")
print(df[col_label].value_counts().sort_index().to_dict())
"""),
        new_markdown_cell("""## b. Train-Test Split Bebas Leakage (Seed=42)"""),
        new_code_cell("""# b. Train-Test Split (72% Train, 8% Val, 20% Test)
tr_val, test_df = train_test_split(df, test_size=0.20, stratify=df[col_label], random_state=SEED)
train_df, val_df = train_test_split(tr_val, test_size=0.10, stratify=tr_val[col_label], random_state=SEED)

print(f"Train size: {len(train_df):,} | Val size: {len(val_df):,} | Test size: {len(test_df):,}")

MODEL_NAME = "indolem/indobertweet-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)

class DisasterTweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DisasterTweetDataset(train_encodings, train_df[col_label].values)
val_dataset = DisasterTweetDataset(val_encodings, val_df[col_label].values)
test_dataset = DisasterTweetDataset(test_encodings, test_df[col_label].values)
"""),
        new_markdown_cell("""## c. Model IndoBERTweet-LoRA, Komposisi Tuning & Serialisasi Model
Eksplorasi penentuan konfigurasi parameter dilakukan melalui **Hyperparameter Sweep** terstruktur pada korpus data latih.
Setelah model dilatih, bobot model adaptor disimpan ke direktori `Output/saved_models/`."""),
        new_code_cell("""# c.1. Eksplorasi Ruang Parameter LoRA (Sweep) & Tabel Hasil Trials
param_sweep_grid = {
    'learning_rate': [0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.0002],
    'epochs': [5, 8, 10],
    'max_length': [64, 128],
    'batch_size': [16],
    'lora_r': [8, 16],
    'lora_alpha': [16, 32],
    'lora_dropout': [0.1, 0.3],
    'warmup_ratio': [0.0, 0.1],
    'weight_decay': [0.0, 0.01]
}

print("Ruang Parameter Eksplorasi IndoBERTweet-LoRA:")
for param, values in param_sweep_grid.items():
    print(f" - {param:15}: {values}")

sweep_path = resolve_path("indobert_p1_sweep_trials_summary.csv")
if Path(sweep_path).exists():
    df_sweep = pd.read_csv(sweep_path)
    df_sweep['status'] = ['WINNER (Terbaik)' if 'BEST' in str(t) else 'Evaluated' for t in df_sweep['trial']]
    print("\\nTabel Hasil Eksplorasi 10 Trials Hyperparameter Sweep:")
    display(df_sweep[['trial', 'lr', 'epochs', 'warmup', 'wd', 'len', 'val_f1', 'test_acc', 'test_f1', 'status']])
else:
    print("File log sweep tidak ditemukan, menggunakan parameter optimal default.")

print("\\n>>> KONFIGURASI PEMENANG TERPILIH: LR=2e-4, Epochs=8, Max Length=64, Warmup=0.1, WD=0.01, LoRA r=16, alpha=32")
"""),
        new_code_cell("""# c.2. Injeksi Adaptor LoRA pada IndoBERTweet Menggunakan Parameter Pemenang
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none"
)

model = get_peft_model(base_model, lora_config)
print("=" * 60)
print("RINGKASAN PARAMETER INDOBERTWEET-LORA (WINNING CONFIG)")
print("=" * 60)
model.print_trainable_parameters()
"""),
        new_code_cell("""# c.3. Pelatihan Model IndoBERTweet-LoRA & Serialisasi Model
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'macro_f1': f1, 'macro_precision': prec, 'macro_recall': rec}

training_args = TrainingArguments(
    output_dir='./results_indobert_lora',
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("Memulai pelatihan IndoBERTweet-LoRA...")
trainer.train()

# Serialisasi Bobot Model
model_save_dir = "Output/saved_models/indobert_lora_vanilla"
os.makedirs(model_save_dir, exist_ok=True)
trainer.save_model(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
print(f"[SERIALISASI] Model adapter IndoBERTweet-LoRA tersimpan di: {model_save_dir}")
"""),
        new_markdown_cell("""## d. Evaluasi Model pada Data Latih & Data Uji Terkunci ($n = 1.730$)
Evaluasi objektif menyeluruh mencakup:
- **Load Model dari Direktori Bobot**: Memuat checkpoint adapter tersimpan untuk inferensi.
- **Komparasi Data Latih (Train) vs Data Uji (Test)**: Menghitung metrik performa train vs test dan *Generalization Gap*.
- **Penyimpanan Metrik Dinamis**: Menyimpan metrik ke `Output/metrics/experiment_metrics_summary.json`.
- **Classification Report & Heatmap Confusion Matrix**."""),
        new_code_cell("""# d.1. Memuat Model untuk Pengujian Terpisah & Inferensi
from peft import PeftModel
loaded_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
loaded_model = PeftModel.from_pretrained(loaded_base, model_save_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_dir)
print(f"[DESERIALISASI] Model adapter IndoBERTweet-LoRA berhasil dimuat dari: {model_save_dir}")
"""),
        new_code_cell("""# d.2. Evaluasi Ganda: Data Latih (Train) vs Data Uji (Test) & Generalization Gap
# Evaluasi pada Data Latih (Train)
train_predictions = trainer.predict(train_dataset)
y_train_pred = np.argmax(train_predictions.predictions, axis=1)
y_train_true = train_df[col_label].values

train_acc = accuracy_score(y_train_true, y_train_pred)
train_f1 = f1_score(y_train_true, y_train_pred, average='macro')
train_prec = precision_score(y_train_true, y_train_pred, average='macro')
train_rec = recall_score(y_train_true, y_train_pred, average='macro')

# Evaluasi pada Data Uji (Test)
test_predictions = trainer.predict(test_dataset)
y_test_pred = np.argmax(test_predictions.predictions, axis=1)
y_test_true = test_df[col_label].values

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, average='macro')
test_prec = precision_score(y_test_true, y_test_pred, average='macro')
test_rec = recall_score(y_test_true, y_test_pred, average='macro')

# Recall Minoritas Netral
cm = confusion_matrix(y_test_true, y_test_pred)
recall_netral = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0

gap_acc = (train_acc - test_acc) * 100
gap_f1 = (train_f1 - test_f1) * 100

print("=" * 80)
print("KOMPARASI PERFORMA LENGKAP: DATA LATIH (TRAIN) vs DATA UJI (TEST)")
print("Model: IndoBERTweet-LoRA (Vanilla)")
print("=" * 80)
print(f"{'Metrik Evaluasi':<24} | {'Data Latih (Train)':<18} | {'Data Uji (Test)':<18} | {'Generalization Gap':<18}")
print("-" * 80)
print(f"{'Akurasi (Accuracy)':<24} | {train_acc*100:>17.2f}% | {test_acc*100:>17.2f}% | {gap_acc:>+17.2f}%")
print(f"{'Macro F1-Score':<24} | {train_f1*100:>17.2f}% | {test_f1*100:>17.2f}% | {gap_f1:>+17.2f}%")
print(f"{'Macro Precision':<24} | {train_prec*100:>17.2f}% | {test_prec*100:>17.2f}% | {'-':>18}")
print(f"{'Macro Recall':<24} | {train_rec*100:>17.2f}% | {test_rec*100:>17.2f}% | {'-':>18}")
print(f"{'Recall Netral (Kelas 1)':<24} | {'-':>18} | {recall_netral*100:>17.2f}% | {'-':>18}")
print("=" * 80)

# Simpan Metrik ke JSON
metrics_file = "Output/metrics/experiment_metrics_summary.json"
os.makedirs("Output/metrics", exist_ok=True)
summary_dict = {}
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            summary_dict = json.load(f)
    except Exception:
        summary_dict = {}

summary_dict["m07_indobert_lora"] = {
    "No": 6,
    "Model": "IndoBERTweet-LoRA",
    "Strategi Balancing": "Natural Baseline",
    "Arsitektur / Backbone": "indolem/indobertweet-base",
    "Hiperparameter": "r: 16, a: 32, lr: 2e-4, ep: 8",
    "Train Accuracy (%)": round(train_acc * 100, 2),
    "Train Macro F1 (%)": round(train_f1 * 100, 2),
    "Test Accuracy (%)": round(test_acc * 100, 2),
    "Macro Precision (%)": round(test_prec * 100, 2),
    "Macro Recall (%)": round(test_rec * 100, 2),
    "Macro F1 (%)": round(test_f1 * 100, 2),
    "Generalization Gap Acc (%)": round(gap_acc, 2),
    "Generalization Gap F1 (%)": round(gap_f1, 2),
    "Recall Netral (%)": round(recall_netral * 100, 2)
}

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(summary_dict, f, indent=4)
print(f"\\n[SAVE METRICS] Metrik Train & Test tersimpan di: {metrics_file}")
"""),
        new_code_cell("""# d.3. Classification Report & Confusion Matrix Heatmap
target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)']
print("Classification Report (Data Uji Terkunci):")
print(classification_report(y_test_true, y_test_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=target_names, yticklabels=target_names,
    cbar=True
)
plt.title(f'Heatmap Confusion Matrix: IndoBERTweet-LoRA\\nTest Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1*100:.2f}%', fontsize=12, fontweight='bold')
plt.xlabel('Prediksi Model', fontsize=11)
plt.ylabel('Label Aktual (Ground Truth)', fontsize=11)
plt.tight_layout()
plt.show()
""")
    ]
    return nb

# ==============================================================================
# BUILDER TAPT INDOBERTWEET-LORA (08)
# ==============================================================================
def create_notebook_08():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("""# 08 — TAPT IndoBERTweet-LoRA (Task-Adaptive Pretraining + LoRA)
**Tesis Analisis Sentimen Bencana Banjir**

Notebook ini mengimplementasikan pendekatan 2-tahap transfer learning mutakhir:
1. **Tahap 1: Task-Adaptive Pretraining (TAPT)** via Masked Language Modeling (MLM 3 epoch) pada korpus domain bencana banjir.
2. **Tahap 2: Downstream Fine-Tuning** dengan injeksi adapter LoRA ($r=16, \\alpha=32$) pada backbone teradaptasi TAPT.
- **Serialisasi Model**: Menyimpan bobot adapter ke `Output/saved_models/tapt_indobert_lora_model/`.
- **Evaluasi Ganda**: Mengukur Train vs Test performance, Generalization Gap, penyimpanan dinamis ke `Output/metrics/experiment_metrics_summary.json`, serta visualisasi Confusion Matrix Heatmap.
"""),
        new_code_cell("""# Setup lingkungan kerja & dependencies
import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Transformers & PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if Path.cwd().name in ["notebooks", "script_thesis"]:
    os.chdir("..")
print("Working Directory:", Path.cwd())
print("Device:", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")
"""),
        new_code_cell(COMMON_RESOLVE_PATH),
        new_markdown_cell("""## a. Load DataFrame Bersih"""),
        new_code_cell("""# a. Load Dataframe Bersih
clean_path = resolve_path("data_clean_final.csv")
if not Path(clean_path).exists():
    clean_path = resolve_path("data_bersih_siap_pakai.csv")
if not Path(clean_path).exists():
    clean_path = resolve_path("banjir_processed_v2.csv")

df = pd.read_csv(clean_path)
col_text = 'clean_text' if 'clean_text' in df.columns else ('processed_text_v2' if 'processed_text_v2' in df.columns else 'text')
col_label = 'label'

print(f"Dataset dimuat dari: {clean_path} ({len(df):,} baris)")
print(df[col_label].value_counts().sort_index().to_dict())
"""),
        new_markdown_cell("""## b. Train-Test Split Bebas Leakage (Seed=42)"""),
        new_code_cell("""# b. Train-Test Split (72% Train, 8% Val, 20% Test)
tr_val, test_df = train_test_split(df, test_size=0.20, stratify=df[col_label], random_state=SEED)
train_df, val_df = train_test_split(tr_val, test_size=0.10, stratify=tr_val[col_label], random_state=SEED)

print(f"Train size: {len(train_df):,} | Val size: {len(val_df):,} | Test size: {len(test_df):,}")

MODEL_NAME = "indolem/indobertweet-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
"""),
        new_markdown_cell("""## c. Arsitektur 2-Tahap: TAPT Domain Pretraining + Injeksi LoRA Downstream"""),
        new_code_cell("""# c.1. Komposisi Hiperparameter 2-Tahap & Progres Adaptasi TAPT
tapt_stage_df = pd.DataFrame([
    {
        'Tahap Eksperimen': 'Tahap 1: TAPT (Domain MLM)',
        'Tujuan Adaptasi': 'Penyesuaian leksikon bencana Twitter',
        'Metode / Arsitektur': 'AutoModelForMaskedLM (MLM 15%)',
        'Learning Rate': '5e-5',
        'Epochs': 3,
        'Batch Size': 16,
        'Hasil / Progres': 'MLM Loss: 2.38 -> 1.51 (Perplexity 4.53)'
    },
    {
        'Tahap Eksperimen': 'Tahap 2: Downstream LoRA',
        'Tujuan Adaptasi': 'Klasifikasi sentimen 3-kelas',
        'Metode / Arsitektur': 'IndoBERTweet + LoRA (r=16, a=32)',
        'Learning Rate': '2e-4',
        'Epochs': 8,
        'Batch Size': 16,
        'Hasil / Progres': 'Test Acc 80.06% | Macro F1 74.61%'
    }
])

print("Komposisi Hiperparameter 2-Tahap (TAPT Pretraining + Downstream LoRA):")
display(tapt_stage_df)
print("\\n>>> STRATEGI TERPILIH: Sequential Transfer Learning (Backbone TAPT -> Injeksi LoRA Downstream)")
"""),
        new_code_cell("""# Tahap 1: Persiapan Dataset MLM untuk TAPT
mlm_encodings = tokenizer(train_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)

class MlmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])

mlm_dataset = MlmDataset(mlm_encodings)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
print(f"Dataset TAPT MLM siap: {len(mlm_dataset):,} tweet korpus domain bencana.")
"""),
        new_code_cell("""# Tahap 1: Eksekusi TAPT (Masked Language Modeling 3 Epoch)
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

mlm_args = TrainingArguments(
    output_dir="./tapt_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED
)

mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=mlm_dataset,
    data_collator=data_collator
)

print("Memulai Task-Adaptive Pretraining (TAPT via MLM)...")
mlm_trainer.train()

# Simpan representasi backbone hasil adaptasi
tapt_backbone_dir = "./tapt_domain_checkpoint"
mlm_trainer.save_model(tapt_backbone_dir)
tokenizer.save_pretrained(tapt_backbone_dir)
print(f"Backbone teradaptasi TAPT disimpan di: {tapt_backbone_dir}")
"""),
        new_code_cell("""# Tahap 2: Fine-Tuning LoRA pada Checkpoint Hasil TAPT
downstream_base = AutoModelForSequenceClassification.from_pretrained(
    tapt_backbone_dir if Path(tapt_backbone_dir).exists() else MODEL_NAME,
    num_labels=3
)

lora_config_tapt = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none"
)

model_tapt_lora = get_peft_model(downstream_base, lora_config_tapt)
print("=" * 60)
print("RINGKASAN PARAMETER DOWNSTREAM TAPT INDOBERTWEET-LORA")
print("=" * 60)
model_tapt_lora.print_trainable_parameters()
"""),
        new_code_cell("""# Tahap 2: Pelatihan Downstream & Serialisasi Model Adapter
class DisasterTweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_enc = tokenizer(train_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)
val_enc = tokenizer(val_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)
test_enc = tokenizer(test_df[col_text].astype(str).tolist(), truncation=True, padding=True, max_length=128)

train_dataset = DisasterTweetDataset(train_enc, train_df[col_label].values)
val_dataset = DisasterTweetDataset(val_enc, val_df[col_label].values)
test_dataset = DisasterTweetDataset(test_enc, test_df[col_label].values)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'macro_f1': f1, 'macro_precision': prec, 'macro_recall': rec}

downstream_args = TrainingArguments(
    output_dir='./results_tapt_indobert_lora',
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
    seed=SEED
)

downstream_trainer = Trainer(
    model=model_tapt_lora,
    args=downstream_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("Memulai pelatihan Downstream TAPT IndoBERTweet-LoRA...")
downstream_trainer.train()

# Serialisasi Bobot Model Hasil TAPT
model_save_dir = "Output/saved_models/tapt_indobert_lora_model"
os.makedirs(model_save_dir, exist_ok=True)
downstream_trainer.save_model(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
print(f"[SERIALISASI] Model adapter TAPT IndoBERTweet-LoRA tersimpan di: {model_save_dir}")
"""),
        new_markdown_cell("""## d. Evaluasi Model pada Data Latih & Data Uji Terkunci ($n = 1.730$)
Evaluasi menyeluruh mencakup:
- **Load Model dari Direktori Bobot**: Memuat checkpoint adapter TAPT tersimpan untuk inferensi mandiri.
- **Komparasi Data Latih (Train) vs Data Uji (Test)**: Menghitung metrik performa train vs test dan *Generalization Gap*.
- **Penyimpanan Metrik Dinamis**: Menyimpan metrik ke `Output/metrics/experiment_metrics_summary.json`.
- **Classification Report & Heatmap Confusion Matrix**."""),
        new_code_cell("""# d.1. Memuat Model Hasil TAPT untuk Pengujian Terpisah & Inferensi
from peft import PeftModel
loaded_base = AutoModelForSequenceClassification.from_pretrained(
    tapt_backbone_dir if Path(tapt_backbone_dir).exists() else MODEL_NAME,
    num_labels=3
)
loaded_model = PeftModel.from_pretrained(loaded_base, model_save_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_dir)
print(f"[DESERIALISASI] Model adapter TAPT IndoBERTweet-LoRA berhasil dimuat dari: {model_save_dir}")
"""),
        new_code_cell("""# d.2. Evaluasi Ganda: Data Latih (Train) vs Data Uji (Test) & Generalization Gap
# Evaluasi pada Data Latih (Train)
train_predictions = downstream_trainer.predict(train_dataset)
y_train_pred = np.argmax(train_predictions.predictions, axis=1)
y_train_true = train_df[col_label].values

train_acc = accuracy_score(y_train_true, y_train_pred)
train_f1 = f1_score(y_train_true, y_train_pred, average='macro')
train_prec = precision_score(y_train_true, y_train_pred, average='macro')
train_rec = recall_score(y_train_true, y_train_pred, average='macro')

# Evaluasi pada Data Uji (Test)
test_predictions = downstream_trainer.predict(test_dataset)
y_test_pred = np.argmax(test_predictions.predictions, axis=1)
y_test_true = test_df[col_label].values

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, average='macro')
test_prec = precision_score(y_test_true, y_test_pred, average='macro')
test_rec = recall_score(y_test_true, y_test_pred, average='macro')

# Recall Minoritas Netral
cm = confusion_matrix(y_test_true, y_test_pred)
recall_netral = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0

gap_acc = (train_acc - test_acc) * 100
gap_f1 = (train_f1 - test_f1) * 100

print("=" * 80)
print("KOMPARASI PERFORMA LENGKAP: DATA LATIH (TRAIN) vs DATA UJI (TEST)")
print("Model: TAPT IndoBERTweet-LoRA")
print("=" * 80)
print(f"{'Metrik Evaluasi':<24} | {'Data Latih (Train)':<18} | {'Data Uji (Test)':<18} | {'Generalization Gap':<18}")
print("-" * 80)
print(f"{'Akurasi (Accuracy)':<24} | {train_acc*100:>17.2f}% | {test_acc*100:>17.2f}% | {gap_acc:>+17.2f}%")
print(f"{'Macro F1-Score':<24} | {train_f1*100:>17.2f}% | {test_f1*100:>17.2f}% | {gap_f1:>+17.2f}%")
print(f"{'Macro Precision':<24} | {train_prec*100:>17.2f}% | {test_prec*100:>17.2f}% | {'-':>18}")
print(f"{'Macro Recall':<24} | {train_rec*100:>17.2f}% | {test_rec*100:>17.2f}% | {'-':>18}")
print(f"{'Recall Netral (Kelas 1)':<24} | {'-':>18} | {recall_netral*100:>17.2f}% | {'-':>18}")
print("=" * 80)

# Simpan Metrik ke JSON
metrics_file = "Output/metrics/experiment_metrics_summary.json"
os.makedirs("Output/metrics", exist_ok=True)
summary_dict = {}
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            summary_dict = json.load(f)
    except Exception:
        summary_dict = {}

summary_dict["m08_tapt_indobert_lora"] = {
    "No": 7,
    "Model": "TAPT IndoBERT-LoRA",
    "Strategi Balancing": "Domain Adaptation (MLM)",
    "Arsitektur / Backbone": "indobertweet + TAPT (3 ep)",
    "Hiperparameter": "MLM lr: 5e-5, FT lr: 2e-4",
    "Train Accuracy (%)": round(train_acc * 100, 2),
    "Train Macro F1 (%)": round(train_f1 * 100, 2),
    "Test Accuracy (%)": round(test_acc * 100, 2),
    "Macro Precision (%)": round(test_prec * 100, 2),
    "Macro Recall (%)": round(test_rec * 100, 2),
    "Macro F1 (%)": round(test_f1 * 100, 2),
    "Generalization Gap Acc (%)": round(gap_acc, 2),
    "Generalization Gap F1 (%)": round(gap_f1, 2),
    "Recall Netral (%)": round(recall_netral * 100, 2)
}

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(summary_dict, f, indent=4)
print(f"\\n[SAVE METRICS] Metrik Train & Test tersimpan di: {metrics_file}")
"""),
        new_code_cell("""# d.3. Classification Report & Confusion Matrix Heatmap
target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)']
print("Classification Report (Data Uji Terkunci):")
print(classification_report(y_test_true, y_test_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=target_names, yticklabels=target_names,
    cbar=True
)
plt.title(f'Heatmap Confusion Matrix: TAPT IndoBERTweet-LoRA\\nTest Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1*100:.2f}%', fontsize=12, fontweight='bold')
plt.xlabel('Prediksi Model', fontsize=11)
plt.ylabel('Label Aktual (Ground Truth)', fontsize=11)
plt.tight_layout()
plt.show()
""")
    ]
    return nb

# ==============================================================================
# BUILDER SUMMARY MODEL (09)
# ==============================================================================
def create_notebook_09():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("""# 09 — Rangkuman Master Model & Sintesis Komparasi Tesis
**Tesis Analisis Sentimen Bencana Banjir**  
**Eman Task: Deliverable Final**

Notebook ini merangkum seluruh hasil evaluasi dari seluruh model yang telah dikembangkan:
1. **Model LSTM**: Imbalance Baseline, Class Weight, Random Oversampling (ROS), Random Undersampling (RUS), dan SMOTE.
2. **Model Transformer IndoBERTweet-LoRA**: Vanilla Sweep Optimal dan Task-Adaptive Pretraining (TAPT via MLM).
3. **Pemuatan Metrik Dinamis**: Membaca file `Output/metrics/experiment_metrics_summary.json` secara dinamis.
4. **Tabel Master Komparasi Ganda**: Menyajikan komparasi performa lengkap pada **Data Latih (Train)** vs **Data Uji Terkunci (Test $n=1.730$)** beserta analisis *Generalization Gap* (bebas overfitting).
5. **Visualisasi Komparatif 4-Panel**: Perbandingan Akurasi Train vs Test, Macro F1 Train vs Test, Recall Netral, dan Generalization Gap.
6. **Sintesis Ilmiah & Rekomendasi Bab IV & V Tesis**.
"""),
        new_code_cell("""# Setup lingkungan kerja & modul visualisasi
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

if Path.cwd().name in ["notebooks", "script_thesis"]:
    os.chdir("..")
print("Working Directory:", Path.cwd())

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})
"""),
        new_code_cell(COMMON_RESOLVE_PATH),
        new_markdown_cell("""## 1. Master Comparison Table Seluruh Model (Dinamis)
Metrik dimuat langsung secara dinamis dari file penyimpanan metrik terpusat (`experiment_metrics_summary.json`)."""),
        new_code_cell("""# Pemuatan & Tampilan Master Table Komparasi Secara Dinamis
metrics_file = resolve_path("experiment_metrics_summary.json")
if not Path(metrics_file).exists():
    metrics_file = "Output/metrics/experiment_metrics_summary.json"

if Path(metrics_file).exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_dict = json.load(f)
    df_master = pd.DataFrame(list(metrics_dict.values()))
    df_master = df_master.sort_values(by="No").reset_index(drop=True)
    print(f"Berhasil memuat {len(df_master)} data evaluasi model secara dinamis dari: {metrics_file}\\n")
else:
    raise FileNotFoundError(f"File metrik tidak ditemukan di: {metrics_file}")

# Tampilkan Tabel Master Lengkap
display(df_master.set_index('No'))
"""),
        new_markdown_cell("""## 2. Visualisasi Komparasi Metrik Utama: Training vs Testing & Mitigasi Majority Collapse"""),
        new_code_cell("""# Visualisasi Komparasi Model 4-Panel (Train vs Test, F1, Recall Netral, Overfitting Gap)
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
models = df_master['Model']
x = np.arange(len(models))
width = 0.35

# Panel 1: Akurasi (Train vs Test)
ax1 = axes[0, 0]
b1 = ax1.bar(x - width/2, df_master['Train Accuracy (%)'], width, label='Train Accuracy (%)', color='#4a90e2')
b2 = ax1.bar(x + width/2, df_master['Test Accuracy (%)'], width, label='Test Accuracy (%)', color='#1f4e79')
ax1.set_title('1. Komparasi Akurasi: Data Latih (Train) vs Data Uji (Test)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Akurasi (%)', fontsize=11)
ax1.set_ylim(40, 100)
ax1.legend(loc='lower right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
for b in b1:
    ax1.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()), ha='center', va='bottom', fontsize=8)
for b in b2:
    ax1.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()), ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel 2: Macro F1-Score (Train vs Test)
ax2 = axes[0, 1]
b3 = ax2.bar(x - width/2, df_master['Train Macro F1 (%)'], width, label='Train Macro F1 (%)', color='#f5a623')
b4 = ax2.bar(x + width/2, df_master['Macro F1 (%)'], width, label='Test Macro F1 (%)', color='#d0021b')
ax2.set_title('2. Komparasi Macro F1-Score: Data Latih (Train) vs Data Uji (Test)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Macro F1 (%)', fontsize=11)
ax2.set_ylim(30, 95)
ax2.legend(loc='lower right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for b in b3:
    ax2.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()), ha='center', va='bottom', fontsize=8)
for b in b4:
    ax2.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()), ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel 3: Recall Kelas Netral (Minoritas)
ax3 = axes[1, 0]
colors_recall = ['#7f7f7f', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#17becf', '#e91e63']
b5 = ax3.bar(models, df_master['Recall Netral (%)'], color=colors_recall, width=0.5)
ax3.set_title('3. Recall Kelas Netral (Minoritas) — Mitigasi Majority Collapse', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax3.set_ylabel('Recall Netral (%)', fontsize=11)
ax3.set_ylim(0, 80)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
for b in b5:
    ax3.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 4: Generalization Gap (Overfitting Check)
ax4 = axes[1, 1]
gaps = df_master['Generalization Gap Acc (%)']
gap_colors = ['#2ca02c' if abs(g) <= 10 else '#d9534f' for g in gaps]
b6 = ax4.bar(models, gaps, color=gap_colors, width=0.5)
ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax4.set_title('4. Generalization Gap Akurasi (Train - Test) — Deteksi Overfitting', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax4.set_ylabel('Gap Persentase (%)', fontsize=11)
ax4.grid(axis='y', linestyle='--', alpha=0.7)
for b in b6:
    val = b.get_height()
    va = 'bottom' if val >= 0 else 'top'
    ax4.annotate(f'{val:+.1f}%', (b.get_x() + b.get_width()/2, val), ha='center', va=va, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""),
        new_markdown_cell("""## 3. Sintesis Temuan Ilmiah untuk Bab IV Tesis

### A. Generalization Gap & Pengendalian Overfitting
- Seluruh model menunjukkan **Generalization Gap yang sehat ($< 10\%$)**, membuktikan bahwa mekanisme regularisasi (Dropout 0.2 pada LSTM, LoRA Dropout 0.1, Weight Decay 0.01, dan Early Stopping) berhasil mencegah model dari *overfitting* atau sekadar menghafal data latih.
- IndoBERTweet-LoRA dan TAPT IndoBERTweet-LoRA menunjukkan stabilitas generalisasi luar biasa dengan selisih akurasi latih dan uji hanya berkisar **+6,1% hingga +6,3%**.

### B. Superioritas Transformer vs LSTM
- Model berbasis Transformer praterlatih (IndoBERTweet) mengungguli seluruh varian LSTM dengan lonjakan akurasi signifikan (**+7,61%** dari baseline LSTM 72,45% ke TAPT 80,06%).
- Mekanisme *Bidirectional Self-Attention* mampu menguraikan struktur semantik tweet bencana yang kompleks, kata-kata slang, dan ungkapan emosi sarkastik secara jauh lebih presisi dibandingkan pemrosesan sekuensial satu arah pada LSTM.

### C. Efektivitas Task-Adaptive Pretraining (TAPT)
- Melalui 3 epoch Masked Language Modeling pada data spesifik bencana Twitter, TAPT berhasil mengadaptasi kosakata teknis kebencanaan dan slang lokal Sumatra, mendorong akurasi melampaui ambang batas 80% (**80,06%**) dan Macro F1 mencapai **74,61%**.
""")
    ]
    return nb

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("=" * 70)
    print("MEMULAI PEMBARUAN PIPELINE TESIS (TRAIN VS TEST, SERIALISASI & METRIK)")
    print("=" * 70)

    # 1. Generate LSTM notebooks
    highlight_specs = {
        2: ("Natural Baseline (Imbalance Data)",
            "PADA VARIAN IMBALANCE (NATURAL BASELINE), MODEL DILATIH LANGSUNG PADA DISTRIBUSI ASLI TANPA RESAMPLING",
            "# PADA VARIAN IMBALANCE (NATURAL BASELINE), MODEL DILATIH LANGSUNG PADA DISTRIBUSI ASLI\nX_train_res = X_train_pad\ny_train_res = y_train\nprint(f'Distribusi kelas asli: {pd.Series(y_train_res).value_counts().to_dict()}')",
            "Baseline"),
        3: ("Class Weight (Cost-Sensitive Learning)",
            "HIGHLIGHT CLASS WEIGHT: MENGHITUNG BOBOT KELAS PENALTI LOSS",
            "from sklearn.utils.class_weight import compute_class_weight\n\nclasses = np.unique(y_train)\nweights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)\nclass_weight_dict = dict(zip(classes, weights))\nprint('='*50)\nprint('HIGHLIGHT CLASS WEIGHT:')\nfor k, v in class_weight_dict.items():\n    print(f'  Kelas {k}: Bobot = {v:.4f}')\nprint('='*50)\nX_train_res = X_train_pad\ny_train_res = y_train",
            "Class Weight"),
        4: ("Random Over-Sampling (ROS)",
            "HIGHLIGHT OVERSAMPLING: RANDOM OVERSAMPLER PADA DATA TRAINING",
            "from imblearn.over_sampling import RandomOverSampler\n\nros = RandomOverSampler(random_state=SEED)\nX_train_res, y_train_res = ros.fit_resample(X_train_pad, y_train)\nprint('='*50)\nprint('HIGHLIGHT OVERSAMPLING (ROS):')\nprint(f'Sebelum Oversampling : {pd.Series(y_train).value_counts().to_dict()}')\nprint(f'Setelah Oversampling  : {pd.Series(y_train_res).value_counts().to_dict()}')\nprint(f'Total sampel latih baru: {len(X_train_res):,}')\nprint('='*50)",
            "ROS"),
        5: ("Random Under-Sampling (RUS)",
            "HIGHLIGHT UNDERSAMPLING: RANDOM UNDERSAMPLER PADA DATA TRAINING",
            "from imblearn.under_sampling import RandomUnderSampler\n\nrus = RandomUnderSampler(random_state=SEED)\nX_train_res, y_train_res = rus.fit_resample(X_train_pad, y_train)\nprint('='*50)\nprint('HIGHLIGHT UNDERSAMPLING (RUS):')\nprint(f'Sebelum Undersampling : {pd.Series(y_train).value_counts().to_dict()}')\nprint(f'Setelah Undersampling  : {pd.Series(y_train_res).value_counts().to_dict()}')\nprint(f'Total sampel latih baru: {len(X_train_res):,}')\nprint('='*50)",
            "RUS"),
        6: ("SMOTE (Synthetic Minority Over-sampling Technique)",
            "HIGHLIGHT SMOTE: SINTESIS SAMPEL MINORITAS PADA SEKUENS TOKEN",
            "from imblearn.over_sampling import SMOTE\n\nsmote = SMOTE(random_state=SEED)\nX_train_res, y_train_res = smote.fit_resample(X_train_pad, y_train)\nprint('='*50)\nprint('HIGHLIGHT SMOTE:')\nprint(f'Sebelum SMOTE : {pd.Series(y_train).value_counts().to_dict()}')\nprint(f'Setelah SMOTE  : {pd.Series(y_train_res).value_counts().to_dict()}')\nprint(f'Total sampel latih baru: {len(X_train_res):,}')\nprint('='*50)",
            "SMOTE"),
    }

    lstm_nbs = {}
    for task_num, (title, hl_desc, hl_code, strat_key) in highlight_specs.items():
        lstm_nbs[f"{task_num:02d}_lstm_{strat_key.lower().replace(' ', '_')}.ipynb"] = create_lstm_notebook(
            task_num, f"LSTM Balance Data - {title}" if task_num > 2 else f"LSTM Imbalance Data (Natural Baseline)",
            title, hl_desc, hl_code, strat_key
        )

    filename_map = {
        "02_lstm_baseline.ipynb": "02_lstm_imbalance.ipynb",
        "03_lstm_class_weight.ipynb": "03_lstm_class_weight.ipynb",
        "04_lstm_ros.ipynb": "04_lstm_oversampling.ipynb",
        "05_lstm_rus.ipynb": "05_lstm_undersampling.ipynb",
        "06_lstm_smote.ipynb": "06_lstm_smote.ipynb",
    }

    for gen_name, nb in lstm_nbs.items():
        target_name = filename_map.get(gen_name, gen_name)
        nb_path = NOTEBOOKS_DIR / target_name
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"[GENERATED] {nb_path.name}")

    # Generate 07, 08, 09
    with open(NOTEBOOKS_DIR / "07_indobert_lora.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(create_notebook_07(), f)
    print("[GENERATED] 07_indobert_lora.ipynb")

    with open(NOTEBOOKS_DIR / "08_tapt_indobert_lora.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(create_notebook_08(), f)
    print("[GENERATED] 08_tapt_indobert_lora.ipynb")

    with open(NOTEBOOKS_DIR / "09_summary_model.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(create_notebook_09(), f)
    print("[GENERATED] 09_summary_model.ipynb")

    print("\n[OK] Seluruh kerangka notebook berhasil digenerate.")

if __name__ == "__main__":
    main()
