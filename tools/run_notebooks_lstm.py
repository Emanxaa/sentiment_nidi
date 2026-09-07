import nbformat
from nbclient import NotebookClient
from pathlib import Path
import time

nb_path = Path("notebooks/02_lstm_imbalance.ipynb")
print(f"Reading {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)

t0 = time.time()
client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook 02 cells...")
client.execute()
elapsed = time.time() - t0

print(f"Writing back to {nb_path} with outputs (took {elapsed:.1f}s)...")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("[SUCCESS] 02_lstm_imbalance.ipynb successfully executed and pre-rendered!")
