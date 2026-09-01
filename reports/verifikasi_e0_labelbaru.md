# Verifikasi Metrik Evaluasi

- File prediksi : `.kaggle-outputs\e1v8\hasil_e1_test_baseline.csv`
- Kolom y_true  : `label_aktual`
- Kolom y_pred  : `label_prediksi`
- Jumlah sampel : **1730**

## Distribusi

- Aktual   : {0: 937, 1: 302, 2: 491}
- Prediksi : {0: 906, 1: 296, 2: 528}

## Metrik

| Metrik | Nilai |
|---|---|
| Accuracy | 0.782659 |
| Precision Macro | 0.734689 |
| Recall Macro | 0.740341 |
| Macro F1 | 0.737070 |
| Weighted F1 | 0.782860 |
| Baseline mayoritas (acc) | 0.541618 |
| Baseline mayoritas (macro F1) | 0.234221 |

## Status

- **COLLAPSE** : TIDAK

## Classification Report

```
              precision    recall  f1-score   support

     negatif       0.86      0.83      0.85       937
      netral       0.58      0.57      0.58       302
     positif       0.76      0.81      0.79       491

    accuracy                           0.78      1730
   macro avg       0.73      0.74      0.74      1730
weighted avg       0.78      0.78      0.78      1730
```

## Confusion Matrix (baris=aktual, kolom=prediksi)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 781 | 88 | 68 |
| netral | 69 | 173 | 60 |
| positif | 56 | 35 | 400 |
