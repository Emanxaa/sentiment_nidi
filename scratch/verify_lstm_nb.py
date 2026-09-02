import json

with open('notebooks/02_legacy_model_lstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'Bidirectional' in src:
        print(f"Warning cell {i} still has Bidirectional: {src[:100]}")
    if any(k in src for k in ['BAGIAN', 'balance', 'LSTM - SKENARIO', 'build_']):
        print(f"Cell {i} ({cell.get('cell_type')}): {src.splitlines()[0][:70]}")
