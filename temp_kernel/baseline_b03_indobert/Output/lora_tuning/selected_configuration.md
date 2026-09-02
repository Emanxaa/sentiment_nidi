# Milestone B2 — Selected Official Configuration

**Generated on:** `2026-09-02 08:48:45`  
**Hardware:** Kaggle Nvidia Tesla T4 GPU  
**Selection Criterion:** Validation Macro F1 (Gate 2)  

---

## 1. Selected Official Hyperparameters

* **Input Column:** `processed_text_v2` (Formally locked in B1.1)
* **Optimal Learning Rate:** `0.0002`
* **Optimal LoRA Rank ($r$):** `8`
* **Optimal LoRA Alpha ($lpha$):** `16`
* **Optimal LoRA Dropout:** `0.05`
* **Epochs:** `5` (with Early Stopping patience 2)
* **Batch Size:** `16`
* **Warmup Ratio:** `0.1`
* **Weight Decay:** `0.01`

---

## 2. Empirical Performance Metrics

* **Validation Macro F1:** **0.5255** (52.55%)
* **Test Macro F1:** **0.5516** (55.16%)
* **Test Accuracy:** **0.6867** (68.67%)

---

## 3. Comparative Progress Analysis

| Benchmark Stage | Val Macro F1 | Test Macro F1 | Test Accuracy | Delta vs Current Baseline | Remaining Gap to Legacy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current Baseline (B1.1)** | 30.28% | 31.34% | 57.05% | — | 23.10 pp |
| **Best LR Calibration** | 52.55% | 55.16% | 68.67% | +23.82 pp | -0.72 pp |
| **Best LoRA Capacity (Official B2)**| **52.55%** | **55.16%** | **68.67%** | **+23.82 pp** | **-0.72 pp** |
| **Legacy Benchmark (`IB-B03-LEG`)** | — | 54.44% | 68.21% | +23.10 pp | Reference Target |

---

## 4. Empirical Justification

1. **Gate 2 Satisfied:** Validation Macro F1 improved substantially over the baseline (52.55% vs 30.28%), satisfying Decision Gate 2.
2. **Underfitting Alleviation:** Calibrating learning rate to `0.0002` enables low-rank adapter updates to overcome the dominant class gradient slope without requiring full fine-tuning.
3. **Parameter Efficiency:** The winning configuration uses only 8 low-rank dimensions, preserving $>99\%$ of transformer weights frozen.
