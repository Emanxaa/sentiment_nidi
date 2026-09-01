# Verifikasi Metrik Evaluasi

- File prediksi : `.kaggle-outputs\e1v9\hasil_e3_kalibrasi_test.csv`
- Kolom y_true  : `label_aktual`
- Kolom y_pred  : `label_prediksi`
- Jumlah sampel : **1730**

## Distribusi

- Aktual   : {0: 937, 1: 302, 2: 491}
- Prediksi : {0: 858, 1: 370, 2: 502}

## Metrik

| Metrik | Nilai |
|---|---|
| Accuracy | 0.774566 |
| Precision Macro | 0.730993 |
| Recall Macro | 0.753175 |
| Macro F1 | 0.739438 |
| Weighted F1 | 0.779347 |
| Baseline mayoritas (acc) | 0.541618 |
| Baseline mayoritas (macro F1) | 0.234221 |

## Status

- **COLLAPSE** : TIDAK

## Classification Report

```
              precision    recall  f1-score   support

     negatif       0.87      0.80      0.84       937
      netral       0.55      0.67      0.60       302
     positif       0.77      0.79      0.78       491

    accuracy                           0.77      1730
   macro avg       0.73      0.75      0.74      1730
weighted avg       0.79      0.77      0.78      1730
```

## Confusion Matrix (baris=aktual, kolom=prediksi)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 750 | 121 | 66 |
| netral | 52 | 202 | 48 |
| positif | 56 | 47 | 388 |
