# Milestone B1 — IndoBERTweet-LoRA Baseline Report

**Generated on:** `2026-09-02 06:49:03`  
**Model Architecture:** `indolem/indobertweet-base-uncased` with PEFT LoRA  
**LoRA Hyperparameters:** `r=16`, `alpha=32`, `dropout=0.1`, `target_modules=query,value`  
**Training Hyperparameters:** `epochs=5`, `batch_size=16`, `lr=2e-05`, `weight_decay=0.01`, `warmup_ratio=0.1`, `early_stopping_patience=2`  
**Seeds Evaluated:** `42`, `123`, `456` (independent initialization and checkpoints)  

---

## 1. Aggregated Performance Summary (IndoBERTweet-LoRA)

| Metric | Mean | Standard Deviation | Seed 42 | Seed 123 | Seed 456 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.6904** (69.04%) | ±0.0154 | 0.7081 | 0.6827 | 0.6803 |
| **Precision** | **0.6176** | ±0.0878 | 0.6892 | 0.6441 | 0.5196 |
| **Recall** | **0.5589** | ±0.0215 | 0.5837 | 0.5485 | 0.5446 |
| **Macro F1** (Primary) | **0.5172** (51.72%) | ±0.0285 | 0.5487 | 0.5096 | 0.4933 |

---

## 2. Automatic Comparison vs Official LSTM Baseline

Direct empirical comparison against the reference LSTM baseline (Milestone M3, identical dataset split and evaluation protocol):

| Metric | LSTM Baseline | IndoBERT-LoRA | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 72.45% | **69.04%** | **-3.41 pp** |
| **Macro F1** | 64.95% | **51.72%** | **-13.23 pp** |
| **Precision** | 67.28% | **61.76%** | **-5.52 pp** |
| **Recall** | 63.78% | **55.89%** | **-7.89 pp** |

*Note: Deltas are calculated as `IndoBERT-LoRA Mean - LSTM Baseline` in percentage points (pp). Positive indicates superior performance.*

---

## 3. Stability & Seed Sensitivity Analysis

* **Mean Macro F1:** `0.5172` (Std Dev: `±0.0285`)
* **Mean Accuracy:** `0.6904` (Std Dev: `±0.0154`)
* **Best-Performing Seed:** `Seed 42` (Macro F1 = `0.5487`, Accuracy = `0.7081`)
* **Worst-Performing Seed:** `Seed 456` (Macro F1 = `0.4933`, Accuracy = `0.6803`)
* **Performance Range:**
  * Macro F1 Range: `0.0554` (0.4933 – 0.5487)
  * Accuracy Range: `0.0278` (0.6803 – 0.7081)

---

## 4. Per-Seed Runtime & Checkpoint Directory

| Seed | Training Time | Restored Best Checkpoint | Output Artifacts Directory |
| :---: | :---: | :---: | :--- |
| **Seed 42** | 167.67s (2.79m) | `best_model/` | [`Output/indobert_lora/seed42/`](seed42/) |
| **Seed 123** | 170.99s (2.85m) | `best_model/` | [`Output/indobert_lora/seed123/`](seed123/) |
| **Seed 456** | 171.64s (2.86m) | `best_model/` | [`Output/indobert_lora/seed456/`](seed456/) |
| **Total** | **604.31s (10.07m)** | — | [`Output/indobert_lora/`](./) |

---

## 5. Methodological Compliance Verification

1. **Zero Data Leakage**: Tokenization performed strictly after data partitioning. The Test set was held out and never used during training, validation, or model selection.
2. **Stratified Split Integrity**: Partitions strictly match LSTM experiments (Train: 6,226 [72%], Val: 692 [8%], Test: 1,730 [20%]).
3. **PEFT LoRA Parameter Efficiency**: Only attention adapter weights and classification head were updated (< 1% trainable parameters). Base model weights were completely frozen.
4. **No Balancing / No Resampling**: Evaluated on true natural distribution without oversampling, undersampling, SMOTE, or synthetic generation.
5. **Full Multi-Split Artifacts**: Confusion matrices generated for Train, Validation, and Test partitions; learning curves generated for loss and accuracy across all epochs.
