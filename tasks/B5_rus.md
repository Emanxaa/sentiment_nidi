# Milestone B5 — Random Undersampling (RUS)

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* configs/indobert_lora.yaml

This milestone benchmarks Random Undersampling (RUS) on IndoBERTweet-LoRA.

Do not implement oversampling or SMOTE.
Stop after this milestone.

---

## Objective
Evaluate the empirical impact of downsampling majority training samples down to the minority class count on IndoBERTweet-LoRA performance across three seeds (`42`, `123`, `456`).

---

## Inputs
* **Dataset:** `Data/processed/banjir_processed_v2.csv` (`processed_text_v2`)
* **Configuration:** `configs/indobert_lora.yaml`
* **Sampling Policy:** Applied strictly to the Training split.
  * Training samples before RUS: $n=6,226$.
  * Training samples after RUS: $n=3,261$ ($1,087$ per class).
  * Validation ($n=692$) and Test ($n=1,730$) partitions remain completely untouched.

---

## Outputs
Directory: `Output/empirical/indobert_rus/`
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
  * `rus_report.md` (with delta comparison vs B1 baseline and LSTM M6 RUS)

---

## Validation Checklist
- [ ] RUS applied strictly to Train split after partitioning.
- [ ] Validation and Test partitions preserved in natural distribution.
- [ ] Three seeds completed independently with seed-specific undersampling.
- [ ] Best checkpoint restored before test evaluation.
- [ ] All 9 artifact files present in every seed directory.

---

## Stop Condition
Stop immediately after generating `rus_report.md` and verifying the checklist. Do not proceed to B6 automatically.
