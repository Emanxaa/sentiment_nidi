import nbformat
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path("notebooks/01_data_processing.ipynb")
print(f"Reading {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook cells...")
client.execute()

print(f"Writing back to {nb_path} with outputs...")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("[SUCCESS] 01_data_processing.ipynb successfully executed and pre-rendered!")
