
# Project Decisions

## Fixed Decisions

- Dataset canonical: thesis-indobert-processed-data
- Random seed: 42
- GPU: Tesla T4
- Target Accuracy: ≥70%
- Metric utama: Macro F1

## Notebook Policy

- Notebook adalah artifact hasil generate.
- Source of Truth berada di src/.
- Jangan edit notebook generated secara manual.

## Kaggle Policy

- Push dataset hanya jika dataset berubah.
- Satu kernel = satu eksperimen.