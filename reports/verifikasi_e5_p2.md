# Verifikasi Metrik Evaluasi

- File prediksi : `.kaggle-outputs\e5\hasil_e5_test.csv`
- Kolom y_true  : `label_aktual`
- Kolom y_pred  : `label_prediksi`
- Jumlah sampel : **1730**

## Distribusi

- Aktual   : {0: 937, 1: 302, 2: 491}
- Prediksi : {0: 766, 1: 426, 2: 538}

## Metrik

| Metrik | Nilai |
|---|---|
| Accuracy | 0.734104 |
| Precision Macro | 0.695429 |
| Recall Macro | 0.730415 |
| Macro F1 | 0.704117 |
| Weighted F1 | 0.742958 |
| Baseline mayoritas (acc) | 0.541618 |
| Baseline mayoritas (macro F1) | 0.234221 |

## Status

- **COLLAPSE** : TIDAK

## Classification Report

```
              precision    recall  f1-score   support

     negatif       0.88      0.72      0.79       937
      netral       0.48      0.68      0.56       302
     positif       0.72      0.79      0.76       491

    accuracy                           0.73      1730
   macro avg       0.70      0.73      0.70      1730
weighted avg       0.77      0.73      0.74      1730
```

## Confusion Matrix (baris=aktual, kolom=prediksi)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 676 | 171 | 90 |
| netral | 40 | 204 | 58 |
| positif | 50 | 51 | 390 |
