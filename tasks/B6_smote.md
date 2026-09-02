# Milestone B6 — SMOTE Feature Balancing

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* configs/indobert_lora.yaml

This milestone benchmarks Synthetic Minority Over-sampling Technique (SMOTE) on IndoBERTweet hidden representations.

Do not implement simple heuristic balancing.
Stop after this milestone.

---

## Objective
Evaluate the empirical impact of synthesizing minority class feature vectors using SMOTE in the IndoBERT embedding space before passing into the classification head across three seeds (`42`, `123`, `456`).

---

## Inputs
* **Dataset:** `Data/processed/banjir_processed_v2.csv` (`processed_text_v2`)
* **Configuration:** `configs/indobert_lora.yaml`
* **SMOTE Policy:** Applied in the 768-dimensional sentence representation space strictly on the Training split.
  * Validation ($n=692$) and Test ($n=1,730$) partitions remain natural and untouched.

---

## Outputs
Directory: `Output/empirical/indobert_smote/`
* Per-seed subdirectories (`seed42/`, `seed123/`, `seed456/`):
  * `best_model/`
  * `history.csv`
  * `metrics.json`
  * `loss_curve.png` & `accuracy_curve.png`
  * `confusion_train.png`, `confusion_val.png`, `confusion_test.png`
  * `classification_report.csv`
* Aggregated summary:
  * `summary.csv`
  * `summary.json`
  * `smote_report.md` (with delta comparison vs B1 baseline and LSTM M7 SMOTE)

---

## Validation Checklist
- [ ] SMOTE applied strictly to Train split embeddings (zero leakage).
- [ ] Validation and Test partitions evaluated with natural embeddings.
- [ ] Three seeds completed independently.
- [ ] Best checkpoint restored before test evaluation.
- [ ] All 9 artifact files present in every seed directory.

---

## Stop Condition
Stop immediately after generating `smote_report.md` and verifying the checklist. Do not proceed to B7 automatically.
