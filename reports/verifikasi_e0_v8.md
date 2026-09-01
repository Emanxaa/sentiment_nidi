# Verifikasi Metrik Evaluasi

- File prediksi : `.kaggle-outputs\hasil_prediksi_test_indobertweet_lora.csv`
- Kolom y_true  : `label_aktual`
- Kolom y_pred  : `label_prediksi`
- Jumlah sampel : **1730**

## Distribusi

- Aktual   : {0: 969, 1: 293, 2: 468}
- Prediksi : {0: 1013, 1: 192, 2: 525}

## Metrik

| Metrik | Nilai |
|---|---|
| Accuracy | 0.725434 |
| Precision Macro | 0.651853 |
| Recall Macro | 0.633793 |
| Macro F1 | 0.636004 |
| Weighted F1 | 0.714984 |
| Baseline mayoritas (acc) | 0.560116 |
| Baseline mayoritas (macro F1) | 0.239348 |

## Status

- **COLLAPSE** : TIDAK

## Classification Report

```
              precision    recall  f1-score   support

     negatif       0.81      0.84      0.82       969
      netral       0.49      0.32      0.39       293
     positif       0.66      0.74      0.69       468

    accuracy                           0.73      1730
   macro avg       0.65      0.63      0.64      1730
weighted avg       0.71      0.73      0.71      1730
```

## Confusion Matrix (baris=aktual, kolom=prediksi)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 816 | 60 | 93 |
| netral | 110 | 95 | 88 |
| positif | 87 | 37 | 344 |
