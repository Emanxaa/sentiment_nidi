import json

with open('notebooks/03_legacy_model_bilstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip() if c['source'] else ''
    print(f"Cell {i:02d} | {c['cell_type']:8s} | {first_line[:70]}")
