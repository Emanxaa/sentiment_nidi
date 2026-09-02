# Milestone B1 — IndoBERTweet-LoRA Baseline

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* configs/indobert_lora.yaml

This milestone implements the official unweighted IndoBERTweet-LoRA baseline on the natural class distribution.

Do not implement balancing.
Stop after this milestone.

---

## Objective
Build a reproducible IndoBERTweet-LoRA baseline directly comparable to the completed LSTM baseline (Milestone M3). Evaluate across three independent seeds (`42`, `123`, `456`) on Kaggle Tesla T4 GPU.

---

## Inputs
* **Dataset:** `Data/processed/banjir_processed_v2.csv`
* **Text Column:** `processed_text_v2`
* **Label Column:** `label` (0: Negative, 1: Neutral, 2: Positive)
* **Configuration:** `configs/indobert_lora.yaml`
* **Splits:** 80% Train, 20% Test (Stratified); 10% of Train for Validation.

---

## Outputs
Directory: `Output/indobert_lora/`
* Per-seed subdirectories (`seed42/`, `seed123/`, `seed456/`):
  * `best_model/` (LoRA adapter weights & tokenizer)
  * `history.csv` (epoch, train_loss, val_loss, train_acc, val_acc)
  * `metrics.json` (multi-split evaluation metrics & metadata)
  * `loss_curve.png` & `accuracy_curve.png` (300 DPI)
  * `confusion_train.png`, `confusion_val.png`, `confusion_test.png` (300 DPI, order: Negative, Neutral, Positive)
  * `classification_report.csv`
* Aggregated summary:
  * `summary.csv`
  * `summary.json`
  * `indobert_lora_report.md` (with automatic delta comparison vs LSTM baseline: Acc 72.45%, F1 64.95%)

---

## Validation Checklist
- [ ] Raw data untouched (`Data/processed/banjir_processed_v2.csv`).
- [ ] Zero data leakage: Tokenization strictly post-split; Test set held out.
- [ ] LoRA applied correctly ($r=16, \alpha=32, \text{dropout}=0.1, <1\%$ trainable parameters).
- [ ] Base transformer weights 100% frozen (full fine-tuning prohibited).
- [ ] Three seeds completed independently (`42`, `123`, `456`).
- [ ] Best checkpoint restored before test evaluation.
- [ ] All 9 artifact files present in every seed directory.
- [ ] `summary.csv`, `summary.json`, and `indobert_lora_report.md` created and verified.

---

## Stop Condition
Stop immediately after generating the aggregated report and verifying the checklist. Do not proceed to B2 automatically.
