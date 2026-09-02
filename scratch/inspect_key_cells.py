import json

with open('notebooks/03_legacy_model_bilstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i in [1, 3, 5, 8, 15, 16, 19, 28, 33]:
    if i < len(nb['cells']):
        print(f"=== CELL {i} ({nb['cells'][i]['cell_type']}) ===")
        print(''.join(nb['cells'][i]['source'])[:300])
