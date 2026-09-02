# Milestone B2.5 — Phase 2: Split Integrity Forensic Report

**Generated on:** `2026-09-02`  
**Focus:** Verification of train, validation, and test sample isolation, stratification, and index equivalence.

---

## 1. Partition Verification

| Partition | Samples ($n$) | Percentage | Stratification | Random State |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | **6,226** | 72.0% | Stratified by `label` | `seed=42` |
| **Validation Set** | **692** | 8.0% | Stratified by `label` | `seed=42` |
| **Test Set** | **1,730** | 20.0% | Stratified by `label` | `seed=42` |
| **Total** | **8,648** | 100.0% | Stratified | Fixed |

---

## 2. Forensic Cross-Check against Legacy Reference

1. **Test Set Partition Concordance:**
   * In legacy notebook `04_model_indobertweet_lora.ipynb`: `X_test_bert` had shape `(1730,)` with labels `y_test`.
   * In `hasil_prediksi_indobertweet_lora_empiris.csv`: exactly 1,730 test samples.
   * **Verification Result:** Every single sample in the current test set matches the legacy test set with **100.00% index and label concordance**.
2. **Train / Validation Concordance:**
   * In legacy notebook `04_model_indobertweet_lora.ipynb` (Cell 5):
     `train_test_split(X_train_bert, y_train, test_size=0.1, stratify=y_train, random_state=42)`
   * In our modular pipeline:
     `train_test_split(train_val_df, test_size=0.1, stratify=train_val_df['label'], random_state=42)`
   * **Verification Result:** The validation set samples overlap **100.0% (692 of 692 samples identical)**.
3. **Data Leakage Check:**
   * Train, Val, and Test partitions share **0 overlapping indices**.
   * Tokenization was conducted strictly post-split.
