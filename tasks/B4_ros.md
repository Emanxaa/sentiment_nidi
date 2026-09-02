# Milestone B4 — Random Oversampling (ROS)

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* configs/indobert_lora.yaml

This milestone benchmarks Random Oversampling (ROS) on IndoBERTweet-LoRA.

Do not implement undersampling or SMOTE.
Stop after this milestone.

---

## Objective
Evaluate the empirical impact of duplicating minority training samples up to the majority class count on IndoBERTweet-LoRA performance across three seeds (`42`, `123`, `456`).

---

## Inputs
* **Dataset:** `Data/processed/banjir_processed_v2.csv` (`processed_text_v2`)
* **Configuration:** `configs/indobert_lora.yaml`
* **Sampling Policy:** Applied strictly to the Training split.
  * Training samples before ROS: $n=6,226$.
  * Training samples after ROS: $n=10,122$ ($3,374$ per class).
  * Validation ($n=692$) and Test ($n=1,730$) partitions remain completely untouched.

---

## Outputs
Directory: `Output/empirical/indobert_ros/`
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
  * `ros_report.md` (with delta comparison vs B1 baseline and LSTM M5 ROS)

---

## Validation Checklist
- [ ] ROS applied strictly to Train split after partitioning.
- [ ] Validation and Test partitions preserved in natural distribution.
- [ ] Three seeds completed independently with seed-specific resampling.
- [ ] Best checkpoint restored before test evaluation.
- [ ] All 9 artifact files present in every seed directory.

---

## Stop Condition
Stop immediately after generating `ros_report.md` and verifying the checklist. Do not proceed to B5 automatically.
