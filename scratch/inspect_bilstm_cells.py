import json

with open('notebooks/03_legacy_model_bilstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(k in src for k in ['BAGIAN', 'fit(', 'balance', 'buat_skenario']):
        print(f"Cell {i} ({cell.get('cell_type')}): " + src[:80].replace('\n', ' '))
