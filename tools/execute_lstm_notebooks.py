"""
Skrip Eksekusi Live Notebook LSTM (02 s.d. 06)
Menggunakan NotebookClient untuk melatih model, serialisasi pickle,
evaluasi Train vs Test, dan pencatatan metrik.
"""

import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"

lstm_notebooks = [
    "02_lstm_imbalance.ipynb",
    "03_lstm_class_weight.ipynb",
    "04_lstm_oversampling.ipynb",
    "05_lstm_undersampling.ipynb",
    "06_lstm_smote.ipynb",
]

for name in lstm_notebooks:
    nb_path = NOTEBOOKS_DIR / name
    print(f"\n{'='*70}")
    print(f"EKSEKUSI LIVE: {nb_path.name}")
    print(f"{'='*70}")
    
    nb = nbformat.read(nb_path, as_version=4)
    t0 = time.time()
    
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()
    elapsed = time.time() - t0
    
    print(f"Menyimpan hasil eksekusi ke {nb_path.name} (Waktu: {elapsed:.1f} detik)...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[BERHASIL] {name} selesai dieksekusi secara live!")

print("\n" + "="*70)
print("SELURUH NOTEBOOK LSTM (02-06) TELAH SELESAI DIEKSEKUSI SECARA LIVE!")
print("="*70)
