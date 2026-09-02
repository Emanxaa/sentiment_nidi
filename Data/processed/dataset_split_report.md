# Stratified Dataset Split Report - Task 06

Generated at: `2026-09-02 05:07:31`  
Source: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\processed\data_preprocessed_v2.csv`  
Output: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\processed\split_data_v2.pkl`  
Random State: `42`  

---

## 1. Split Distribution Summary

* **Total Samples:** 8,648
* **Train Set:** 6,226 samples (72.0%)
* **Validation Set:** 692 samples (8.0%)
* **Test Set:** 1,730 samples (20.0%)

---

## 2. Label Stratification Table

| Class | Train (6,226) | Validation (692) | Test (1,730) |
| :--- | :--- | :--- | :--- |
| **Negatif (0)** | 3,374 (54.19%) | 375 (54.19%) | 937 (54.16%) |
| **Netral (1)** | 1,087 (17.46%) | 121 (17.49%) | 302 (17.46%) |
| **Positif (2)** | 1,765 (28.35%) | 196 (28.32%) | 491 (28.38%) |

---

## 3. Data Integrity & Reproducibility

* **Stratification Verified:** Exact class proportion matched across all splits.
* **No Leakage:** Strict partition across indices without overlap.
* **Dual Stream Keys Available:** `X_train_bert`, `X_train_lstm`, `y_train`, etc.
