import json

with open('temp_kernel/pull_test_v2/thesis-indobert-v2.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']).strip()
    print(f"Cell {i} ({c.get('cell_type')}): " + src[:80].replace('\n', ' '))
