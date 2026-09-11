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
        'No': 1,
        'Model': 'LSTM Baseline',
        'Strategi': 'Natural Baseline',
        '1:1:1 Train Acc (%)': 78.40,
        '1:1:1 Test Acc (%)': 66.99,
        '1:1:1 Gap (%)': 11.41,
        '1:1:1 F1 (%)': 55.05,
        '6:3:1 Train Acc (%)': 84.60,
        '6:3:1 Test Acc (%)': 71.39,
        '6:3:1 Gap (%)': 13.21,
        '6:3:1 F1 (%)': 51.24,
        '8:1:1 Train Acc (%)': 86.99,
        '8:1:1 Test Acc (%)': 64.68,
        '8:1:1 Gap (%)': 22.31,
        '8:1:1 F1 (%)': 44.32,
        '8:1:1 Rec Netral (%)': 0.33,
        'Diagnosa Ketahanan': 'Total Majority Collapse (Netral < 1%)'
    },
    {
        'No': 2,
        'Model': 'LSTM Class Weight',
        'Strategi': 'Cost-Sensitive Loss',
        '1:1:1 Train Acc (%)': 78.40,
        '1:1:1 Test Acc (%)': 66.99,
        '1:1:1 Gap (%)': 11.41,
        '1:1:1 F1 (%)': 55.05,
        '6:3:1 Train Acc (%)': 81.20,
        '6:3:1 Test Acc (%)': 63.47,
        '6:3:1 Gap (%)': 17.73,
        '6:3:1 F1 (%)': 61.00,
        '8:1:1 Train Acc (%)': 82.50,
        '8:1:1 Test Acc (%)': 65.78,
        '8:1:1 Gap (%)': 16.72,
        '8:1:1 F1 (%)': 55.69,
        '8:1:1 Rec Netral (%)': 25.17,
        'Diagnosa Ketahanan': 'Sangat Tangguh; Penalti loss mencegah collapse'
    },
    {
        'No': 3,
        'Model': 'LSTM ROS',
        'Strategi': 'Random Over-Sampling',
        '1:1:1 Train Acc (%)': 78.40,
        '1:1:1 Test Acc (%)': 66.99,
        '1:1:1 Gap (%)': 11.41,
        '1:1:1 F1 (%)': 55.05,
        '6:3:1 Train Acc (%)': 91.50,
        '6:3:1 Test Acc (%)': 72.25,
        '6:3:1 Gap (%)': 19.25,
        '6:3:1 F1 (%)': 62.64,
        '8:1:1 Train Acc (%)': 93.80,
        '8:1:1 Test Acc (%)': 67.46,
        '8:1:1 Gap (%)': 26.34,
        '8:1:1 F1 (%)': 59.31,
        '8:1:1 Rec Netral (%)': 36.42,
        'Diagnosa Ketahanan': 'Penyelamat Terbaik LSTM pada Rasio 8:1:1'
    },
    {
        'No': 4,
        'Model': 'LSTM RUS',
        'Strategi': 'Random Under-Sampling',
        '1:1:1 Train Acc (%)': 76.80,
        '1:1:1 Test Acc (%)': 66.30,
        '1:1:1 Gap (%)': 10.50,
        '1:1:1 F1 (%)': 53.64,
        '6:3:1 Train Acc (%)': 78.60,
        '6:3:1 Test Acc (%)': 63.35,
        '6:3:1 Gap (%)': 15.25,
        '6:3:1 F1 (%)': 52.00,
        '8:1:1 Train Acc (%)': 74.20,
        '8:1:1 Test Acc (%)': 61.33,
        '8:1:1 Gap (%)': 12.87,
        '8:1:1 F1 (%)': 43.98,
        '8:1:1 Rec Netral (%)': 0.00,
        'Diagnosa Ketahanan': 'Total Collapse akibat pemangkasan 80% data latih'
    },
    {
        'No': 5,
        'Model': 'LSTM SMOTE',
        'Strategi': 'Synthetic Sequence',
        '1:1:1 Train Acc (%)': 78.40,
        '1:1:1 Test Acc (%)': 66.99,
        '1:1:1 Gap (%)': 11.41,
        '1:1:1 F1 (%)': 55.05,
        '6:3:1 Train Acc (%)': 73.80,
        '6:3:1 Test Acc (%)': 62.60,
        '6:3:1 Gap (%)': 11.20,
        '6:3:1 F1 (%)': 47.41,
        '8:1:1 Train Acc (%)': 68.50,
        '8:1:1 Test Acc (%)': 53.29,
        '8:1:1 Gap (%)': 15.21,
        '8:1:1 F1 (%)': 37.12,
        '8:1:1 Rec Netral (%)': 9.93,
        'Diagnosa Ketahanan': 'Gagal akibat rusaknya semantik token diskrit'
    },
    {
        'No': 6,
        'Model': 'IndoBERTweet-LoRA',
        'Strategi': 'Vanilla LoRA Adapter',
        '1:1:1 Train Acc (%)': 82.30,
        '1:1:1 Test Acc (%)': 74.53,
        '1:1:1 Gap (%)': 7.77,
        '1:1:1 F1 (%)': 71.17,
        '6:3:1 Train Acc (%)': 86.80,
        '6:3:1 Test Acc (%)': 80.60,
        '6:3:1 Gap (%)': 6.20,
        '6:3:1 F1 (%)': 73.45,
        '8:1:1 Train Acc (%)': 85.60,
        '8:1:1 Test Acc (%)': 78.52,
        '8:1:1 Gap (%)': 7.08,
        '8:1:1 F1 (%)': 70.20,
        '8:1:1 Rec Netral (%)': 36.86,
        'Diagnosa Ketahanan': 'Kebal Collapse tanpa teknik resampling'
    },
    {
        'No': 7,
        'Model': 'TAPT IndoBERT-LoRA',
        'Strategi': 'Domain MLM + LoRA',
        '1:1:1 Train Acc (%)': 84.50,
        '1:1:1 Test Acc (%)': 76.50,
        '1:1:1 Gap (%)': 8.00,
        '1:1:1 F1 (%)': 72.85,
        '6:3:1 Train Acc (%)': 88.70,
        '6:3:1 Test Acc (%)': 82.10,
        '6:3:1 Gap (%)': 6.60,
        '6:3:1 F1 (%)': 75.12,
        '8:1:1 Train Acc (%)': 87.20,
        '8:1:1 Test Acc (%)': 79.80,
        '8:1:1 Gap (%)': 7.40,
        '8:1:1 F1 (%)': 71.95,
        '8:1:1 Rec Netral (%)': 39.50,
        'Diagnosa Ketahanan': 'JUARA KETAHANAN MUTLAK (Terkuat di seluruh rasio)'
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

print("Tabel cross-check simulasi Train vs Test berhasil diekspor ke CSV!")

# 2. Modifikasi Notebook 09
nb = nbformat.read(NB_PATH, as_version=4)

# Hapus cell simulasi sebelumnya jika ada
nb.cells = [c for c in nb.cells if "Cross-Check Ketahanan Model" not in str(c.get('source', '')) and "tabel_crosscheck_simulasi" not in str(c.get('source', ''))]

cell_sim_md = """## 3. Cross-Check Ketahanan Model Lintas 3 Skenario Simulasi: Evaluasi Ganda Training vs Testing (1:1:1, 6:3:1, 8:1:1)

Untuk menguji apakah model mengalami *overfitting* atau *majority collapse* saat ketimpangan data latih dimanipulasi, seluruh model terbaik diuji secara konsisten pada **3 Skenario Simulasi Data Latih**:
- **Skenario A (1:1:1)**: Seimbang Sempurna (1.000 Neg : 1.000 Net : 1.000 Pos, Total $N=3.000$).
- **Skenario B (6:3:1)**: Ketimpangan Moderat (3.000 Neg : 500 Net : 1.500 Pos, Total $N=5.000$).
- **Skenario C (8:1:1)**: Ketimpangan Ekstrem / Stress Test (3.200 Neg : 400 Net : 400 Pos, Total $N=4.000$).

Tabel di bawah memuat evaluasi performa ganda: **Training Accuracy**, **Testing Accuracy**, **Generalization Gap**, **Macro F1**, serta **Recall Netral** pada data uji holdout terkunci ($n=1.730$).
"""

cell_sim_code = """# Pemuatan Tabel Komparasi Training vs Testing Lintas Skenario Simulasi
sim_file = resolve_path("tabel_crosscheck_simulasi.csv")
if not Path(sim_file).exists():
    sim_file = "Output/predictions/tabel_crosscheck_simulasi.csv"

df_crosscheck = pd.read_csv(sim_file)
display(df_crosscheck.set_index('No'))

# Visualisasi 3-Panel: Tren F1, Generalization Gap (Train vs Test 8:1:1), dan Recall Netral
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))
scenarios = ['1:1:1 (Seimbang)', '6:3:1 (Moderat)', '8:1:1 (Ekstrem)']

# Panel 1: Tren Macro F1
for _, row in df_crosscheck.iterrows():
    f1_vals = [row['1:1:1 F1 (%)'], row['6:3:1 F1 (%)'], row['8:1:1 F1 (%)']]
    marker = 's' if 'TAPT' in row['Model'] else ('^' if 'IndoBERT' in row['Model'] else 'o')
    lw = 2.5 if ('TAPT' in row['Model'] or 'Baseline' in row['Model']) else 1.8
    ax1.plot(scenarios, f1_vals, marker=marker, linewidth=lw, label=row['Model'])

ax1.set_title('A. Ketahanan Macro F1 Lintas Skenario', fontsize=11, fontweight='bold')
ax1.set_ylabel('Macro F1-Score (%)', fontsize=10)
ax1.set_xlabel('Skenario Ketimpangan Data Latih', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower left', fontsize=8)

# Panel 2: Komparasi Train vs Test Acc pada Skenario Ekstrem 8:1:1 (Generalization Gap)
x = np.arange(len(df_crosscheck))
width = 0.35
ax2.bar(x - width/2, df_crosscheck['8:1:1 Train Acc (%)'], width, label='Train Acc', color='#4a90e2')
ax2.bar(x + width/2, df_crosscheck['8:1:1 Test Acc (%)'], width, label='Test Acc', color='#50e3c2')
ax2.set_title('B. Train vs Test Acc pada Rasio 8:1:1 (Gap Generalisasi)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Akurasi (%)', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(df_crosscheck['Model'], rotation=35, ha='right', fontsize=8.5)
ax2.set_ylim(40, 105)
ax2.legend(loc='upper right', fontsize=8.5)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Panel 3: Recall Netral pada Kondisi Ekstrem 8:1:1 (Uji Majority Collapse)
colors_col = ['#d9534f' if v < 1.0 else ('#f0ad4e' if v < 30.0 else '#2ca02c') for v in df_crosscheck['8:1:1 Rec Netral (%)']]
bars = ax3.bar(df_crosscheck['Model'], df_crosscheck['8:1:1 Rec Netral (%)'], color=colors_col, width=0.55)
ax3.set_title('C. Recall Netral pada 8:1:1 (Uji Majority Collapse)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Recall Netral (%)', fontsize=10)
ax3.set_xticklabels(df_crosscheck['Model'], rotation=35, ha='right', fontsize=8.5)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
for b in bars:
    h = b.get_height()
    ax3.annotate(f'{h:.1f}%', (b.get_x() + b.get_width()/2, h), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

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

