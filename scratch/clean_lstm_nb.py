import json
import re

with open('notebooks/02_legacy_model_lstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    full_text = "".join(cell['source'])
    # Replace Bidirectional multi-line
    full_text = re.sub(r'Bidirectional\(\s*LSTM\((\d+)\)\s*\)', r'LSTM(\1)', full_text)
    full_text = full_text.replace(", Bidirectional", "").replace("Bidirectional, ", "")
    # Split back to lines preserving newlines
    lines = [line + "\n" for line in full_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines.pop()
    cell['source'] = lines

with open('notebooks/02_legacy_model_lstm.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Berhasil membersihkan Bidirectional dari 02_legacy_model_lstm.ipynb")
