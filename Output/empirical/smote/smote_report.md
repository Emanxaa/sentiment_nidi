# Milestone M7 — SMOTE (Synthetic Minority Oversampling) Report

**Generated on:** `2026-09-02 13:58:23`  
**Architecture:** PyTorch LSTM (`Units=128`, `Embedding=128`, `Dropout=0.3`)  
**Balancing Method:** Sequence-Level SMOTE on Train Split ($6,226 \rightarrow 10,122$ samples)  
**Seeds Evaluated:** `42`, `123`, `456`  

---

## 1. Class Distribution Analysis

### Before SMOTE (Original Train Partition)
| Class | Count | Percentage |
| :--- | :---: | :---: |
| **Negative (0)** | 3,374 | 54.19% |
| **Neutral (1)** | 1,087 | 17.46% |
| **Positive (2)** | 1,765 | 28.35% |
| **Total** | **6,226** | **100.00%** |

### After SMOTE (Balanced Train Partition)
| Class | Count | Percentage | Synthetic Samples Added |
| :--- | :---: | :---: | :---: |
| **Negative (0)** | 3,374 | 33.33% | 0 (Majority anchor) |
| **Neutral (1)** | 3,374 | 33.33% | +2,287 samples |
| **Positive (2)** | 3,374 | 33.33% | +1,609 samples |
| **Total** | **10,122** | **100.00%** | **+3,896 samples** |

---

## 2. Multi-Experiment Performance Comparison (All 5 Methods)

| Metric | Baseline (M3) | Class Weight (M4) | ROS (M5) | RUS (M6) | SMOTE (M7) | Δ vs Baseline | Std Dev ($\pm\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7245 (72.45%) | 0.6592 (65.92%) | 0.6705 (67.05%) | 0.6295 (62.95%) | **0.4272** (42.72%) | **-0.2973** (-29.73 pp) | ±0.0176 |
| **Macro F1** (Primary) | 0.6495 (64.95%) | 0.6270 (62.70%) | 0.6271 (62.71%) | 0.5892 (58.92%) | **0.4076** (40.76%) | **-0.2419** (-24.19 pp) | ±0.0307 |
| **Precision** | 0.6728 (67.28%) | 0.6312 (63.12%) | 0.6325 (63.25%) | 0.5945 (59.45%) | **0.4491** (44.91%) | **-0.2237** (-22.37 pp) | ±0.0290 |
| **Recall** | 0.6378 (63.78%) | 0.6515 (65.15%) | 0.6373 (63.73%) | 0.6030 (60.30%) | **0.4445** (44.45%) | **-0.1933** (-19.33 pp) | ±0.0181 |

---

## 3. Empirical Interpretation & Stability Analysis

* **Macro F1 Impact:** SMOTE achieved a mean Macro F1 of 40.76% (-24.19 pp vs baseline). Synthetic integer interpolation in discrete token sequence space generated slight semantic noise compared to exact text replication in ROS.
* **Recall vs Precision:** Macro Recall was 44.45% (-19.33 pp vs baseline), and Precision was 44.91%.
* **Comparison Across Balancing Methods:**
  $$\text{Baseline (64.95\%)} > \text{ROS (62.71\%)} \approx \text{Class Weight (62.70\%)} > \text{RUS (58.92\%)} > \text{SMOTE (40.76\%)} $$
* **Best-Performing Seed:** `Seed 123` (Macro F1 = `0.4382`, Accuracy = `0.4474`)
* **Worst-Performing Seed:** `Seed 42` (Macro F1 = `0.3768`, Accuracy = `0.4185`)
* **Macro F1 Range:** `0.0614` (0.3768 – 0.4382)
* **Variability Assessment:** **High (Significant Seed Sensitivity)**

---

## 4. Per-Seed Performance Breakdown

| Seed | Accuracy | Precision | Recall | Macro F1 | Val Macro F1 | Training Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.4185 | 0.4182 | 0.4308 | 0.3768 | 0.3863 | 76.36s |
| **Seed 123** | 0.4474 | 0.4757 | 0.4650 | 0.4382 | 0.4434 | 76.67s |
| **Seed 456** | 0.4156 | 0.4535 | 0.4376 | 0.4079 | 0.4272 | 72.32s |
