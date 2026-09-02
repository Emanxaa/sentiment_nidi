# Milestone M6 — Random Undersampling (RUS) Report

**Generated on:** `2026-09-02 13:51:53`  
**Architecture:** PyTorch LSTM (`Units=128`, `Embedding=128`, `Dropout=0.3`)  
**Balancing Method:** Random Undersampling on Train Split ($6,226 \rightarrow 3,261$ samples)  
**Seeds Evaluated:** `42`, `123`, `456`  

---

## 1. Class Distribution Analysis

### Before Random Undersampling (Original Train Partition)
| Class | Count | Percentage |
| :--- | :---: | :---: |
| **Negative (0)** | 3,374 | 54.19% |
| **Neutral (1)** | 1,087 | 17.46% |
| **Positive (2)** | 1,765 | 28.35% |
| **Total** | **6,226** | **100.00%** |

### After Random Undersampling (Balanced Train Partition)
| Class | Count | Percentage | Reduction Count |
| :--- | :---: | :---: | :---: |
| **Negative (0)** | 1,087 | 33.33% | -2,287 samples |
| **Neutral (1)** | 1,087 | 33.33% | 0 samples (Minority baseline) |
| **Positive (2)** | 1,087 | 33.33% | -678 samples |
| **Total** | **3,261** | **100.00%** | **-2,965 samples (-47.62%)** |

---

## 2. Multi-Experiment Performance Comparison

| Metric | Baseline (M3) | Class Weight (M4) | ROS (M5) | RUS (M6) | Δ vs Baseline | Std Dev ($\pm\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7245 (72.45%) | 0.6592 (65.92%) | 0.6705 (67.05%) | **0.6295** (62.95%) | **-0.0950** (-9.50 pp) | ±0.0360 |
| **Macro F1** (Primary) | 0.6495 (64.95%) | 0.6270 (62.70%) | 0.6271 (62.71%) | **0.5892** (58.92%) | **-0.0603** (-6.03 pp) | ±0.0233 |
| **Precision** | 0.6728 (67.28%) | 0.6312 (63.12%) | 0.6325 (63.25%) | **0.5945** (59.45%) | **-0.0783** (-7.83 pp) | ±0.0171 |
| **Recall** | 0.6378 (63.78%) | 0.6515 (65.15%) | 0.6373 (63.73%) | **0.6030** (60.30%) | **-0.0348** (-3.48 pp) | ±0.0210 |

---

## 3. Empirical Interpretation & Stability Analysis

* **Macro F1 Impact:** RUS achieved 58.92% Macro F1 (-6.03 pp vs baseline). Discarding 47.6% of training data reduced sample diversity, impacting generalization.
* **Recall Impact:** Macro Recall was 60.30% (-3.48 pp vs baseline).
* **Efficiency:** Training on 3,261 samples significantly reduced runtime per epoch while maintaining stable convergence.
* **Best-Performing Seed:** `Seed 123` (Macro F1 = `0.6155`, Accuracy = `0.6705`)
* **Worst-Performing Seed:** `Seed 456` (Macro F1 = `0.5709`, Accuracy = `0.6150`)
* **Macro F1 Range:** `0.0446` (0.5709 – 0.6155)
* **Variability Assessment:** **Moderate (Acceptable Variance)**

---

## 4. Per-Seed Performance Breakdown

| Seed | Accuracy | Precision | Recall | Macro F1 | Val Macro F1 | Training Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.6029 | 0.5961 | 0.6063 | 0.5813 | 0.5774 | 53.67s |
| **Seed 123** | 0.6705 | 0.6108 | 0.6222 | 0.6155 | 0.6079 | 41.51s |
| **Seed 456** | 0.6150 | 0.5767 | 0.5805 | 0.5709 | 0.5457 | 34.03s |
