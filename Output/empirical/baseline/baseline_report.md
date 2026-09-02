# Milestone M3 — Baseline Final Report (Three-Seed Reproducibility)

**Generated on:** `2026-09-02 13:16:43`  
**Architecture:** PyTorch LSTM (`Units=128`, `Embedding=128`, `Dropout=0.3`)  
**Optimizer:** Adam (`lr=0.0005`, `batch_size=16`, `patience=3`)  
**Seeds Evaluated:** `42`, `123`, `456`  

---

## 1. Aggregated Performance Summary

| Metric | Mean | Standard Deviation | Seed 42 | Seed 123 | Seed 456 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.7245** (72.45%) | ±0.0107 | 0.7231 | 0.7358 | 0.7145 |
| **Precision** | **0.6728** | ±0.0209 | 0.6701 | 0.6950 | 0.6534 |
| **Recall** | **0.6378** | ±0.0212 | 0.6507 | 0.6493 | 0.6133 |
| **Macro F1** (Primary) | **0.6495** (64.95%) | ±0.0213 | 0.6578 | 0.6654 | 0.6253 |

---

## 2. Stability Analysis

* **Mean Macro F1:** `0.6495`
* **Macro F1 Std Dev:** `0.0213`
* **Mean Accuracy:** `0.7245`
* **Accuracy Std Dev:** `0.0107`
* **Best-Performing Seed:** `Seed 123` (Macro F1 = `0.6654`, Accuracy = `0.7358`)
* **Worst-Performing Seed:** `Seed 456` (Macro F1 = `0.6253`, Accuracy = `0.7145`)
* **Performance Range:**
  * Macro F1 Range: `0.0401` (0.6253 – 0.6654)
  * Accuracy Range: `0.0213` (0.7145 – 0.7358)
* **Variability Assessment:** **Moderate (Acceptable Baseline Variance)**

---

## 3. Per-Seed Runtime & Checkpoints

| Seed | Training Time (s) | Best Checkpoint | Output Directory |
| :---: | :---: | :---: | :--- |
| **Seed 42** | 117.91s | `best_model.pt` | [`Output/empirical/baseline/seed42/`](seed42/) |
| **Seed 123** | 124.11s | `best_model.pt` | [`Output/empirical/baseline/seed123/`](seed123/) |
| **Seed 456** | 93.41s | `best_model.pt` | [`Output/empirical/baseline/seed456/`](seed456/) |
| **Total** | **335.43s** | — | — |

---

## 4. Methodological Compliance

1. **Strict Zero Leakage**: Tokenizer fitted exclusively on the 80% Training partition for each seed.
2. **Stratification Preserved**: Class proportions (`Negative`, `Neutral`, `Positive`) matched across Train (72%), Val (8%), and Test (20%).
3. **Reproducibility**: Entire pipeline reproducible via deterministically seeded CLI executions.
4. **Reference Baseline Established**: Serves as the official empirical baseline against which future balancing techniques (Milestone M4+) will be benchmarked.
