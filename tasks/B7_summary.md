# Milestone B7 — IndoBERT vs LSTM Comprehensive Thesis Synthesis

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* Output/empirical/ (all LSTM and IndoBERT experiment directories)

This milestone performs final cross-model synthesis, statistical significance testing, and thesis report compilation.

Do not retrain any models.
Stop after this milestone.

---

## Objective
Synthesize all empirical results across LSTM (M3-M7) and IndoBERTweet-LoRA (B1-B6). Conduct McNemar tests for statistical significance, compute Cohen's Kappa for inter-model agreement, and generate publication-quality Chapter IV thesis comparison artifacts.

---

## Inputs
* LSTM outputs: `Output/empirical/baseline/`, `class_weight/`, `random_oversampling/`, `random_undersampling/`, `smote/`
* IndoBERT outputs: `Output/empirical/indobert_{baseline, class_weight, ros, rus, smote}/`
* Test Ground Truth: 1,730 test labels and predictions from best checkpoints.

---

## Outputs
Directory: `Output/summary/`
* `master_thesis_comparison.csv` (consolidated Accuracy, Macro F1, Precision, Recall across all 10 experimental configurations)
* `statistical_significance_mcnemar.csv` (pairwise McNemar contingency tables and p-values)
* `cohen_kappa_agreement.csv` (inter-model agreement coefficients)
* `comparison_macro_f1.png` & `comparison_accuracy.png` (publication-ready 300 DPI comparative bar charts)
* `thesis_chapter_4_report.md` (comprehensive Indonesian research findings formatted for thesis Chapter IV)

---

## Validation Checklist
- [ ] All 10 experimental configurations included.
- [ ] Means and standard deviations accurately aggregated across 3 seeds.
- [ ] McNemar test computed on paired test predictions ($p < 0.05$ threshold).
- [ ] No model retraining triggered (read-only synthesis).
- [ ] All comparison charts generated at 300 DPI.

---

## Stop Condition
Stop immediately after generating `thesis_chapter_4_report.md` and verifying all synthesis artifacts.
