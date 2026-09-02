# Milestone M5 — Random Oversampling (ROS) Report

**Generated on:** `2026-09-02 13:40:49`  
**Architecture:** PyTorch LSTM (`Units=128`, `Embedding=128`, `Dropout=0.3`)  
**Balancing Method:** Random Oversampling on Train Split ($6,226 \rightarrow 10,122$ samples)  
**Seeds Evaluated:** `42`, `123`, `456`  

---

## 1. Class Distribution Analysis

### Before Random Oversampling (Original Train Partition)
| Class | Count | Percentage |
| :--- | :---: | :---: |
| **Negative (0)** | 3,374 | 54.19% |
| **Neutral (1)** | 1,087 | 17.46% |
| **Positive (2)** | 1,765 | 28.35% |
| **Total** | **6,226** | **100.00%** |

### After Random Oversampling (Balanced Train Partition)
| Class | Count | Percentage | Duplication Factor |
| :--- | :---: | :---: | :---: |
| **Negative (0)** | 3,374 | 33.33% | 1.00x |
| **Neutral (1)** | 3,374 | 33.33% | 3.10x |
| **Positive (2)** | 3,374 | 33.33% | 1.91x |
| **Total** | **10,122** | **100.00%** | — |

---

## 2. Multi-Experiment Performance Comparison

| Metric | Baseline (M3) | Class Weight (M4) | ROS (M5) | Δ vs Baseline | Std Dev ($\pm\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7245 (72.45%) | 0.6592 (65.92%) | **0.6705** (67.05%) | **-0.0540** (-5.40 pp) | ±0.0231 |
| **Macro F1** (Primary) | 0.6495 (64.95%) | 0.6270 (62.70%) | **0.6271** (62.71%) | **-0.0224** (-2.24 pp) | ±0.0116 |
| **Precision** | 0.6728 (67.28%) | 0.6312 (63.12%) | **0.6325** (63.25%) | **-0.0403** (-4.03 pp) | ±0.0132 |
| **Recall** | 0.6378 (63.78%) | 0.6515 (65.15%) | **0.6373** (63.73%) | **-0.0005** (-0.05 pp) | ±0.0045 |

---

## 3. Empirical Interpretation & Stability Analysis

* **Macro F1 Impact:** ROS showed a slight trade-off in Macro F1 compared to Baseline, but outperformed Class Weight.
* **Recall Impact:** ROS maintained comparable Macro Recall with the baseline.
* **Precision & Accuracy Impact:** Accuracy shifted by -5.40 pp, while Precision shifted by -4.03 pp.
* **Best-Performing Seed:** `Seed 123` (Macro F1 = `0.6389`, Accuracy = `0.6942`)
* **Worst-Performing Seed:** `Seed 456` (Macro F1 = `0.6158`, Accuracy = `0.6480`)
* **Macro F1 Range:** `0.0231` (0.6158 – 0.6389)
* **Variability Assessment:** **Moderate (Acceptable Variance)**

---

## 4. Per-Seed Performance Breakdown

| Seed | Accuracy | Precision | Recall | Macro F1 | Val Macro F1 | Training Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.6694 | 0.6433 | 0.6333 | 0.6267 | 0.6036 | 131.48s |
| **Seed 123** | 0.6942 | 0.6365 | 0.6422 | 0.6389 | 0.6227 | 122.43s |
| **Seed 456** | 0.6480 | 0.6178 | 0.6363 | 0.6158 | 0.6277 | 109.97s |
