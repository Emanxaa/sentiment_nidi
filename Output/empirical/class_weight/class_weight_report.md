# Milestone M4 — Class Weight Experiment Report

**Generated on:** `2026-09-02 13:30:56`  
**Architecture:** PyTorch LSTM (`Units=128`, `Embedding=128`, `Dropout=0.3`)  
**Loss Function:** Weighted Cross-Entropy (`CrossEntropyLoss(weight=class_weights)`)  
**Seeds Evaluated:** `42`, `123`, `456`  

---

## 1. Computed Class Weights

Balanced class weights computed strictly from the Training partition ($N=6,226$ samples):

* **Negative (0):** `0.615096` (Support: `3,374`)
* **Neutral (1):** `1.909230` (Support: `1,087`)
* **Positive (2):** `1.175826` (Support: `1,765`)

---

## 2. Aggregated Performance & Comparison Against Baseline

| Metric | Baseline (M3) | Class Weight (M4) | Delta | Std Dev ($\pm\sigma$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7245 (72.45%) | **0.6592** (65.92%) | **-0.0653** (-6.53 pp) | ±0.0360 |
| **Macro F1** (Primary) | 0.6495 (64.95%) | **0.6270** (62.70%) | **-0.0225** (-2.25 pp) | ±0.0201 |
| **Precision** | 0.6728 | **0.6312** | **-0.0416** | ±0.0119 |
| **Recall** | 0.6378 | **0.6515** | **+0.0137** | ±0.0013 |

---

## 3. Stability & Seed Analysis

* **Mean Macro F1:** `0.6270`
* **Macro F1 Std Dev:** `0.0201`
* **Best-Performing Seed:** `Seed 123` (Macro F1 = `0.6449`, Accuracy = `0.6931`)
* **Worst-Performing Seed:** `Seed 42` (Macro F1 = `0.6052`, Accuracy = `0.6214`)
* **Performance Range (Macro F1):** `0.0397` (0.6052 – 0.6449)
* **Variability Assessment:** **Moderate (Acceptable Variance)**

---

## 4. Per-Seed Performance Breakdown

| Seed | Accuracy | Precision | Recall | Macro F1 | Val Macro F1 | Training Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.6214 | 0.6184 | 0.6501 | 0.6052 | 0.5903 | 103.69s |
| **Seed 123** | 0.6931 | 0.6420 | 0.6518 | 0.6449 | 0.6396 | 95.59s |
| **Seed 456** | 0.6630 | 0.6332 | 0.6526 | 0.6310 | 0.6288 | 94.56s |
