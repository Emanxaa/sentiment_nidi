# Verifikasi Metrik Evaluasi

- File prediksi : `.kaggle-outputs\e4\hasil_e4_test.csv`
- Kolom y_true  : `label_aktual`
- Kolom y_pred  : `label_prediksi`
- Jumlah sampel : **1730**

## Distribusi

- Aktual   : {0: 937, 1: 302, 2: 491}
- Prediksi : {0: 852, 1: 336, 2: 542}

## Metrik

| Metrik | Nilai |
|---|---|
| Accuracy | 0.771098 |
| Precision Macro | 0.725574 |
| Recall Macro | 0.745942 |
| Macro F1 | 0.733935 |
| Weighted F1 | 0.774263 |
| Baseline mayoritas (acc) | 0.541618 |
| Baseline mayoritas (macro F1) | 0.234221 |

## Status

- **COLLAPSE** : TIDAK

## Classification Report

```
              precision    recall  f1-score   support

     negatif       0.87      0.79      0.83       937
      netral       0.57      0.63      0.60       302
     positif       0.74      0.81      0.77       491

    accuracy                           0.77      1730
   macro avg       0.73      0.75      0.73      1730
weighted avg       0.78      0.77      0.77      1730
```

## Confusion Matrix (baris=aktual, kolom=prediksi)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 744 | 106 | 87 |
| netral | 57 | 190 | 55 |
| positif | 51 | 40 | 400 |
