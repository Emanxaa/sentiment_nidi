# Verifikasi Metrik Evaluasi

- File prediksi : `.kaggle-outputs/p1_ft_sweep/exp_p1_ft_sweep_test.csv`
- Kolom y_true  : `label_aktual`
- Kolom y_pred  : `label_prediksi`
- Jumlah sampel : **1730**

## Distribusi

- Aktual   : {0: 937, 1: 302, 2: 491}
- Prediksi : {0: 866, 1: 328, 2: 536}

## Metrik

| Metrik | Nilai |
|---|---|
| Accuracy | 0.779769 |
| Precision Macro | 0.731785 |
| Recall Macro | 0.748831 |
| Macro F1 | 0.739033 |
| Weighted F1 | 0.782423 |
| Baseline mayoritas (acc) | 0.541618 |
| Baseline mayoritas (macro F1) | 0.234221 |

## Status

- **COLLAPSE** : TIDAK

## Classification Report

```
              precision    recall  f1-score   support

     negatif       0.88      0.81      0.84       937
      netral       0.56      0.61      0.59       302
     positif       0.75      0.82      0.79       491

    accuracy                           0.78      1730
   macro avg       0.73      0.75      0.74      1730
weighted avg       0.79      0.78      0.78      1730
```

## Confusion Matrix (baris=aktual, kolom=prediksi)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 760 | 108 | 69 |
| netral | 54 | 185 | 63 |
| positif | 52 | 35 | 404 |
