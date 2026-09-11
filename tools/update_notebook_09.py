import os
import sys
import json
import time
import shutil
from pathlib import Path
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(".").resolve()
NB_PATH = ROOT / "notebooks/09_summary_model.ipynb"
SCRIPT_THESIS_NB = ROOT / "script_thesis/09_summary_model.ipynb"

print(f"Reading {NB_PATH}...")
nb = nbformat.read(NB_PATH, as_version=4)

# 1. Update Cell 4 to also export df_master to CSV files
cell_4_code = """# Pemuatan & Tampilan Master Table Komparasi Secara Dinamis
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

# Ekspor tabel master ke direktori Output
for out_path in [
    Path("Output/predictions/tabel_komparasi_seluruh_model.csv"),
    Path("Output/summary/master_summary.csv"),
    Path("Output/summary/final_results_table.csv"),
]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(out_path, index=False)
    print(f"[SAVE] Tabel master tersimpan di: {out_path}")
"""
nb.cells[4].source = cell_4_code

# 2. Update Cell 7 (Sintesis Temuan Ilmiah)
cell_7_md = """## 3. Sintesis Temuan Ilmiah untuk Bab IV Tesis

### A. Generalization Gap & Pengendalian Overfitting
- Seluruh model menunjukkan **Generalization Gap yang sangat sehat ($< 10\\%$)**, membuktikan bahwa mekanisme regularisasi (Dropout 0.2 pada LSTM, LoRA Dropout 0.1, Weight Decay 0.01, dan Early Stopping) berhasil mencegah model dari *overfitting* atau sekadar menghafal data latih.
- IndoBERTweet-LoRA dan TAPT IndoBERTweet-LoRA menunjukkan stabilitas generalisasi luar biasa dengan selisih akurasi latih dan uji hanya berkisar **+6,17% hingga +6,34%**.

### B. Superioritas Transformer vs LSTM
- Model berbasis Transformer praterlatih (IndoBERTweet) mengungguli seluruh varian LSTM dengan lonjakan akurasi signifikan (**+9,14%** dari baseline LSTM 70,92% ke TAPT 80,06%) serta lonjakan Macro F1 sebesar **+13,05%** (dari 61,56% ke 74,61%).
- Mekanisme *Bidirectional Self-Attention* mampu menguraikan struktur semantik tweet bencana yang kompleks, kata-kata slang, dan ungkapan emosi sarkastik secara jauh lebih presisi dibandingkan pemrosesan sekuensial satu arah pada LSTM.

### C. Efektivitas Task-Adaptive Pretraining (TAPT)
- Melalui 3 epoch Masked Language Modeling (MLM) pada data spesifik bencana Twitter, TAPT berhasil mengadaptasi kosakata teknis kebencanaan dan slang lokal Sumatra, mendorong akurasi melampaui ambang batas 80% (**80,06%**) dan Macro F1 mencapai **74,61%**.
"""
nb.cells[7].source = cell_7_md

# 3. Add or update Cell 8 (Pembahasan Ilmiah Khusus Imbalance Data)
cell_8_md = """## 4. Pembahasan Ilmiah: Mengapa Imbalanced Data Alami Mencapai Akurasi Lebih Tinggi dari Beberapa Teknik Balancing?

Temuan empiris menunjukkan bahwa **LSTM Baseline (Natural Imbalance)** mencatat akurasi global **70,92%**, yang **lebih tinggi** dibandingkan teknik penyeimbangan seperti **SMOTE (64,39%)** dan **Random Undersampling (53,06%)**.

Fenomena ini dijelaskan oleh 3 prinsip fundamental dalam teori *Machine Learning*:

1. **The Accuracy Paradox (Ilusi Akurasi & Keruntuhan Kelas Mayoritas)**:
   - Pada dataset tweet banjir ini, kelas mayoritas (Positif & Negatif) mendominasi $\\approx 82\\%$ dari seluruh data uji, sedangkan kelas minoritas Netral hanya $\\approx 18\\%$.
   - Model imbalanced secara alami bias memprediksi kelas mayoritas. Dengan menebak mayoritas hampir sepanjang waktu, model mudah mencatat skor akurasi global tinggi. Namun, model menderita **Majority Collapse** dengan Recall Netral hanya **28,81%** (gagal mendeteksi informasi penting yang bersifat netral).

2. **Dampak Pembuangan Data pada Random Undersampling (*Information Loss*)**:
   - Untuk menyeimbangkan rasio kelas, RUS membuang lebih dari 45% data mayoritas (Negatif dan Positif).
   - Pada data teks, pemotongan ini membuang ribuan variasi kata penting, konteks tata bahasa, dan ungkapan bencana lokal. Akibatnya, representasi kosakata model menjadi sangat miskin dan akurasi anjlok drastis ke **53,06%**.

3. **Kerusakan Semantik pada SMOTE (*Discrete Token Corruption*)**:
   - SMOTE dirancang untuk fitur angka kontinu (seperti data tabular numerik) melalui interpolasi linier antar titik sampel ($x_{new} = x_i + \\lambda (x_{zi} - x_i)$).
   - Pada representasi sekuens teks (deretan ID integer kata pada embedding), interpolasi matematika membangkitkan token sintetis acak yang tidak memiliki arti leksikal dalam kamus bahasa Indonesia. Token artifisial ini mengacaukan mekanisme pemrosesan gerbang memori LSTM, menurunkan akurasi ke **64,39%**.

4. **Kapan Teknik Penyeimbangan Dikatakan Sukses?**:
   - Tujuan penyeimbangan data **bukan untuk mendongkrak akurasi global**, melainkan **menyelamatkan kelas minoritas**.
   - Teknik seperti **Class Weight** dan **Random Oversampling (ROS)** sukses mendongkrak Recall Netral dari **28,81%** menjadi **40,40%** dan **48,68%**, memberikan sensitivitas deteksi bencana yang jauh lebih adil tanpa merusak struktur token bahasa.
"""

if len(nb.cells) <= 8:
    nb.cells.append(new_markdown_cell(cell_8_md))
else:
    nb.cells[8].source = cell_8_md

# Save changes before execution
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Executing {NB_PATH} with NotebookClient...")
t0 = time.time()
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()
elapsed = time.time() - t0
print(f"Executed successfully in {elapsed:.1f} seconds!")

# Save executed notebook
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

# Copy to script_thesis/
shutil.copy2(NB_PATH, SCRIPT_THESIS_NB)
print(f"Copied to {SCRIPT_THESIS_NB}")
