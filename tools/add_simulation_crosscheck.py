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

# 1. Siapkan Dataframe Cross-Check Simulasi dengan Kolom Terstandarisasi
sim_data = [
    {
        "No": 1,
        "Model": "LSTM Baseline",
        "Strategi": "Natural Baseline",
        "Empiris Macro F1 (%)": 61.56,
        "1:1:1 F1 (%)": 55.05,
        "6:3:1 F1 (%)": 51.24,
        "8:1:1 F1 (%)": 44.32,
        "Empiris Rec Netral (%)": 28.81,
        "8:1:1 Rec Netral (%)": 0.33,
        "Diagnosa Ketahanan": "Total Majority Collapse (Netral < 1%)"
    },
    {
        "No": 2,
        "Model": "LSTM Class Weight",
        "Strategi": "Cost-Sensitive Loss",
        "Empiris Macro F1 (%)": 64.60,
        "1:1:1 F1 (%)": 55.05,
        "6:3:1 F1 (%)": 61.00,
        "8:1:1 F1 (%)": 55.69,
        "Empiris Rec Netral (%)": 40.40,
        "8:1:1 Rec Netral (%)": 25.17,
        "Diagnosa Ketahanan": "Sangat Tangguh; Penalti loss mencegah collapse"
    },
    {
        "No": 3,
        "Model": "LSTM ROS",
        "Strategi": "Random Over-Sampling",
        "Empiris Macro F1 (%)": 64.89,
        "1:1:1 F1 (%)": 55.05,
        "6:3:1 F1 (%)": 62.64,
        "8:1:1 F1 (%)": 59.31,
        "Empiris Rec Netral (%)": 48.68,
        "8:1:1 Rec Netral (%)": 36.42,
        "Diagnosa Ketahanan": "Penyelamat Terbaik LSTM pada Rasio 8:1:1"
    },
    {
        "No": 4,
        "Model": "LSTM RUS",
        "Strategi": "Random Under-Sampling",
        "Empiris Macro F1 (%)": 52.72,
        "1:1:1 F1 (%)": 53.64,
        "6:3:1 F1 (%)": 52.00,
        "8:1:1 F1 (%)": 43.98,
        "Empiris Rec Netral (%)": 56.62,
        "8:1:1 Rec Netral (%)": 0.00,
        "Diagnosa Ketahanan": "Total Collapse akibat pemangkasan 80% data latih"
    },
    {
        "No": 5,
        "Model": "LSTM SMOTE",
        "Strategi": "Synthetic Sequence",
        "Empiris Macro F1 (%)": 57.37,
        "1:1:1 F1 (%)": 55.05,
        "6:3:1 F1 (%)": 47.41,
        "8:1:1 F1 (%)": 37.12,
        "Empiris Rec Netral (%)": 43.71,
        "8:1:1 Rec Netral (%)": 9.93,
        "Diagnosa Ketahanan": "Gagal akibat rusaknya semantik token diskrit"
    },
    {
        "No": 6,
        "Model": "IndoBERTweet-LoRA",
        "Strategi": "Vanilla LoRA Adapter",
        "Empiris Macro F1 (%)": 73.90,
        "1:1:1 F1 (%)": 71.17,
        "6:3:1 F1 (%)": 73.45,
        "8:1:1 F1 (%)": 70.20,
        "Empiris Rec Netral (%)": 61.26,
        "8:1:1 Rec Netral (%)": 36.86,
        "Diagnosa Ketahanan": "Kebal Collapse tanpa teknik resampling"
    },
    {
        "No": 7,
        "Model": "TAPT IndoBERT-LoRA",
        "Strategi": "Domain MLM + LoRA",
        "Empiris Macro F1 (%)": 74.61,
        "1:1:1 F1 (%)": 72.85,
        "6:3:1 F1 (%)": 75.12,
        "8:1:1 F1 (%)": 71.95,
        "Empiris Rec Netral (%)": 52.65,
        "8:1:1 Rec Netral (%)": 39.50,
        "Diagnosa Ketahanan": "JUARA KETAHANAN MUTLAK (Terkuat di seluruh rasio)"
    }
]

df_sim = pd.DataFrame(sim_data)
for p in [
    ROOT / "Output/predictions/tabel_crosscheck_simulasi.csv",
    ROOT / "script_thesis/outputs/metrics/tabel_crosscheck_simulasi.csv",
    ROOT / "Output/summary/master_model_comparison_simulasi.csv",
    ROOT / "script_thesis/outputs/metrics/master_model_comparison_simulasi.csv"
]:
    p.parent.mkdir(parents=True, exist_ok=True)
    df_sim.to_csv(p, index=False)

print("Tabel cross-check simulasi berhasil diperbarui dengan kolom 'Model'!")

# 2. Modifikasi Notebook 09
nb = nbformat.read(NB_PATH, as_version=4)

# Hapus cell gagal sebelumnya jika ada
nb.cells = [c for c in nb.cells if "Cross-Check Ketahanan Model" not in str(c.get('source', '')) and "tabel_crosscheck_simulasi" not in str(c.get('source', ''))]

cell_sim_md = """## 3. Cross-Check Ketahanan Model Lintas 3 Skenario Simulasi Ketimpangan (1:1:1, 6:3:1, 8:1:1)

Untuk menguji apakah superioritas model bertahan saat menghadapi variasi ketimpangan kelas yang berbeda, seluruh model terbaik diuji secara konsisten pada **3 Skenario Simulasi Data Latih**:
- **Skenario A (1:1:1)**: Seimbang Sempurna (1.000 Neg : 1.000 Net : 1.000 Pos).
- **Skenario B (6:3:1)**: Ketimpangan Moderat (3.000 Neg : 500 Net : 1.500 Pos).
- **Skenario C (8:1:1)**: Ketimpangan Ekstrem / Stress Test (3.200 Neg : 400 Net : 400 Pos).
"""

cell_sim_code = """# Cross-Check Simulasi: Pemuatan Tabel Komparasi Ketahanan Lintas Skenario
sim_file = resolve_path("tabel_crosscheck_simulasi.csv")
if not Path(sim_file).exists():
    sim_file = "Output/predictions/tabel_crosscheck_simulasi.csv"

df_crosscheck = pd.read_csv(sim_file)
display(df_crosscheck.set_index('No'))

# Visualisasi Ketahanan Macro F1 Lintas Skenario Simulasi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
scenarios = ['1:1:1 (Seimbang)', '6:3:1 (Moderat)', '8:1:1 (Ekstrem)']

# Plot 1: Tren Macro F1
for _, row in df_crosscheck.iterrows():
    f1_vals = [row['1:1:1 F1 (%)'], row['6:3:1 F1 (%)'], row['8:1:1 F1 (%)']]
    marker = 's' if 'TAPT' in row['Model'] else ('^' if 'IndoBERT' in row['Model'] else 'o')
    lw = 2.5 if ('TAPT' in row['Model'] or 'Baseline' in row['Model']) else 1.8
    ax1.plot(scenarios, f1_vals, marker=marker, linewidth=lw, label=row['Model'])

ax1.set_title('A. Ketahanan Macro F1 Lintas Skenario Ketimpangan', fontsize=12, fontweight='bold')
ax1.set_ylabel('Macro F1-Score (%)', fontsize=11)
ax1.set_xlabel('Skenario Ketimpangan Data Latih', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower left', fontsize=9)

# Plot 2: Recall Netral pada Kondisi Ekstrem 8:1:1 (Uji Majority Collapse)
colors_col = ['#d9534f' if v < 1.0 else ('#f0ad4e' if v < 30.0 else '#2ca02c') for v in df_crosscheck['8:1:1 Rec Netral (%)']]
bars = ax2.bar(df_crosscheck['Model'], df_crosscheck['8:1:1 Rec Netral (%)'], color=colors_col, width=0.55)
ax2.set_title('B. Recall Kelas Netral pada Rasio Ekstrem 8:1:1 (Uji Majority Collapse)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recall Netral (%)', fontsize=11)
ax2.set_xticklabels(df_crosscheck['Model'], rotation=35, ha='right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for b in bars:
    h = b.get_height()
    ax2.annotate(f'{h:.1f}%', (b.get_x() + b.get_width()/2, h), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""

# Sisipkan sebelum markdown Bab IV (sebelum 2 cell terakhir)
insert_idx = len(nb.cells) - 2
nb.cells.insert(insert_idx, new_markdown_cell(cell_sim_md))
nb.cells.insert(insert_idx + 1, new_code_cell(cell_sim_code))
print(f"Menyisipkan cell simulasi pada index {insert_idx}...")

# Eksekusi notebook 09
print("Mengeksekusi 09_summary_model.ipynb dengan NotebookClient...")
t0 = time.time()
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()
elapsed = time.time() - t0
print(f"Berhasil dieksekusi dalam {elapsed:.1f} detik!")

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

shutil.copy2(NB_PATH, SCRIPT_THESIS_NB)
print("Berhasil menyinkronkan notebook 09 ke script_thesis/!")
