import nbformat
from nbclient import NotebookClient
from pathlib import Path
import time

lstm_notebooks = [
    "02_lstm_imbalance.ipynb",
    "03_lstm_class_weight.ipynb",
    "04_lstm_oversampling.ipynb",
    "05_lstm_undersampling.ipynb",
    "06_lstm_smote.ipynb",
]

for name in lstm_notebooks:
    nb_path = Path("notebooks") / name
    print(f"\n{'='*60}\nReading and executing {nb_path}...\n{'='*60}")
    nb = nbformat.read(nb_path, as_version=4)
    t0 = time.time()
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    elapsed = time.time() - t0
    print(f"Writing back to {nb_path} with outputs (took {elapsed:.1f}s)...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[SUCCESS] {name} executed and pre-rendered!")

print("\nALL REMAINING LSTM NOTEBOOKS EXECUTED SUCCESSFULLY!")
