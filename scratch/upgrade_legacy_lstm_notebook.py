import json
import copy

with open('notebooks/03_legacy_model_bilstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nb_lstm = copy.deepcopy(nb)

for cell in nb_lstm['cells']:
    new_source = []
    for line in cell['source']:
        # Replace titles and text
        l = line
        l = l.replace('biLSTM', 'LSTM').replace('BiLSTM', 'LSTM').replace('bilstm', 'lstm')
        l = l.replace('Bidirectional(LSTM(64))', 'LSTM(64)')
        l = l.replace('Bidirectional(LSTM(units))', 'LSTM(units)')
        l = l.replace('Bidirectional(LSTM(', 'LSTM(')
        # Fix double brackets if any
        l = l.replace('model_lstm_emp.add(\n    Bidirectional(\n        LSTM(64)\n    )\n)', 'model_lstm_emp.add(\n    LSTM(64)\n)')
        new_source.append(l)
    cell['source'] = new_source

with open('notebooks/02_legacy_model_lstm.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_lstm, f, indent=1, ensure_ascii=False)

print("Berhasil meng-upgrade notebooks/02_legacy_model_lstm.ipynb sehingga memiliki seluruh 5 varian lengkap persis seperti BiLSTM!")
