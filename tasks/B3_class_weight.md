# Milestone B3 — Class Weight Balancing

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* configs/indobert_lora.yaml (or optimal params from B2)

This milestone benchmarks cost-sensitive loss weighting on IndoBERTweet-LoRA.

Do not implement sampling or SMOTE.
Stop after this milestone.

---

## Objective
Evaluate whether weighting the CrossEntropy loss function inversely proportional to class frequencies improves minority class recall and Macro F1 on IndoBERTweet-LoRA across three independent seeds (`42`, `123`, `456`).

---

## Inputs
* **Dataset:** `Data/processed/banjir_processed_v2.csv` (`processed_text_v2`)
* **Configuration:** `configs/indobert_lora.yaml`
* **Class Weight Formula:** $w_j = \frac{N}{C \times n_j}$ computed strictly from the Training partition ($n=6,226$).
  * $N$: Total training samples.
  * $C$: Number of classes (3).
  * $n_j$: Number of training samples in class $j$.

---

## Outputs
Directory: `Output/empirical/indobert_class_weight/`
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
  * `class_weight_report.md` (with delta comparison vs unweighted B1 baseline and LSTM M4 class weight)

---

## Validation Checklist
- [ ] Weights calculated strictly from Training split (zero leakage into Val/Test).
- [ ] Loss function: `nn.CrossEntropyLoss(weight=class_weights)`.
- [ ] Validation and Test loss computed without weights (unweighted evaluation).
- [ ] Three seeds completed independently.
- [ ] Best checkpoint restored before test evaluation.
- [ ] All 9 artifact files present in every seed directory.

---

## Stop Condition
Stop immediately after generating `class_weight_report.md` and verifying the checklist. Do not proceed to B4 automatically.
