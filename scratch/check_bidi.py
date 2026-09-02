import json

with open('notebooks/02_legacy_model_lstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i in [3, 7, 11, 19, 28, 33]:
    print(f"=== Cell {i} ===")
    for line in nb['cells'][i]['source']:
        if 'Bidirectional' in line or 'model' in line:
            print("  ", line.strip())
