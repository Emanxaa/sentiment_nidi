# Milestone B2.5 — Phase 5: Training Pipeline Forensic Report

**Generated on:** `2026-09-02`  
**Focus:** Comparison of model architecture, optimization hyperparameters, training dynamics, and checkpoint restoration.

---

## 1. Architectural & Training Parameters Comparison

| Component | Legacy Historical (`IB-B03-LEG`) | Milestone B2 Tuned Baseline | Classification |
| :--- | :--- | :--- | :---: |
| **Base Model** | `indolem/indobertweet-base-uncased` | `indolem/indobertweet-base-uncased` | Identical |
| **Base Dropout** | `hidden_dropout_prob=0.3`, `attention_probs_dropout_prob=0.3` | `0.1` (HuggingFace Default) | Moderate |
| **PEFT LoRA Config** | $r=16, lpha=32, 	ext{dropout}=0.3$ | $r=8, lpha=16, 	ext{dropout}=0.05$ | Moderate |
| **Optimizer** | AdamW (`weight_decay=0.01`) | AdamW (`weight_decay=0.01`) | Identical |
| **Learning Rate** | `0.0002` (`2e-4`) | `0.0002` (`2e-4`) | Identical |
| **Batch Size** | `16` | `16` | Identical |
| **Epochs** | `5` | `5` | Identical |
| **Warmup Ratio** | `0.0` (Default unconfigured) | `0.1` | Moderate |
| **Early Stopping** | None (Trained all 5 epochs) | `EarlyStoppingCallback(patience=2)` | Moderate |
| **Checkpoint Restoration** | `load_best_model_at_end=True` (**Restored Epoch 2: checkpoint-780**) | `load_best_model_at_end=True` | **CRITICAL** |

---

## 2. Checkpoint Restoration Forensic Finding

* Inspection of `baseline/B03_indobert/best_indobertweet_lora_empiris/checkpoint-780/trainer_state.json` proved:
  * The historical model reached its peak performance at **Epoch 2.0 (Step 780)**.
  * Validation Macro F1 peaked at **0.6977 (69.77%)** and Validation Accuracy at **0.7572 (75.72%)**.
  * By Epoch 4 and 5, training loss continued to drop, but validation macro F1 stagnated.
  * Because `load_best_model_at_end=True` was active with `metric_for_best_model="f1_macro"`, `checkpoint-780` was loaded for test evaluation, delivering **73.45% Test Macro F1**!
