
# Kaggle Thesis Experiment Agent

## Goal

Reproduce and improve LSTM, BiLSTM, and IndoBERT-LoRA experiments for the thesis.

## Working Scope

Allowed:
- src/
- scripts/
- configs/
- notebooks/
- experiments/

Never read:
- Data/
- Output/
- __pycache__/
- .ipynb_checkpoints/
- *.png
- *.jpg
- *.csv (except metrics.csv)

## Experiment Rules

- Work on only one experiment at a time.
- Read only one config file (E01–E08).
- Never modify unrelated notebooks.
- Generated notebooks are artifacts, not the source of truth.

## Workflow

1. Read config.
2. Patch or generate notebook.
3. Create kernel-metadata.json.
4. Push to Kaggle.
5. Wait until completion.
6. Download metrics.
7. Update experiments/results.csv.
8. check for error every 30 mins and update it. 

## Success Criteria

- Test accuracy >70%.
- Balanced methods should outperform baseline.

# Kaggle Push Rules

Notebook mapping:

* `b-01` → `username/b-01-lstm` (CPU)
* `b-02` → `username/b-02-bilstm` (CPU)
* `b-03` → `username/b-03-indobertlora` (GPU)

Current status:

* `b-01` sedang berjalan. Jangan menghentikan, restart, atau push versi baru ke `b-01` sampai ada instruksi eksplisit.

Rules:

1. Edit hanya notebook yang diminta (`b-01`, `b-02`, atau `b-03`).
2. Sebelum push, `cd` ke folder notebook tersebut.
3. Jalankan `kaggle kernels push` hanya untuk notebook yang diminta.
4. Jangan mengubah `kernel-metadata.json` (terutama field `id`).
5. Sebutkan target kernel sebelum melakukan push.
