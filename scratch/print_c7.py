import json

with open('notebooks/02_legacy_model_lstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(''.join(nb['cells'][7]['source']))
