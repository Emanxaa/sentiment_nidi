# Milestone B2 — LoRA Hyperparameter & Learning Rate Tuning

Before starting, read:
* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/INDOBERT_LORA.md
* Output/audit/indobert_failure_analysis.md
* configs/indobert_lora.yaml

This milestone executes systematic hyperparameter tuning to identify the optimal configuration for IndoBERTweet-LoRA on disaster sentiment classification.

Do not implement balancing yet.
Stop after this milestone.

---

## Objective
Identify the optimal combination of LoRA rank ($r$), alpha ($\alpha$), dropout, and learning rate on the Validation set to resolve underfitting and maximize Validation Macro F1.

---

## Inputs
* **Dataset:** `Data/processed/banjir_processed_v2.csv` (`processed_text_v2`)
* **Base Configuration:** `configs/indobert_lora.yaml`
* **Evaluation Split:** Validation Set only ($n=692$). Test Set ($n=1,730$) remains completely held out and untouched.
* **Search Space:**
  * Rank ($r$): `[8, 16]`
  * Alpha ($\alpha$): `[16, 32]`
  * Dropout: `[0.1, 0.3]`
  * Learning Rate: `[2e-5, 5e-5, 1e-4, 2e-4, 3e-4]`

---

## Outputs
Directory: `Output/hparam_search_lora/`
* `tuning_results.csv` (full grid search table with trial parameters, Val Loss, Val Accuracy, Val Macro F1)
* `best_params.yaml` (optimal parameter configuration extracted for downstream milestones)
* `lr_response_curve.png` (300 DPI plot illustrating Macro F1 as a function of learning rate)
* `tuning_report.md` (summary of optimal configuration, parameter sensitivity analysis, and rationale)

---

## Validation Checklist
- [ ] Test set strictly held out (zero leakage during search).
- [ ] All trials evaluated strictly on the Validation partition.
- [ ] Base transformer weights frozen in all trials.
- [ ] `tuning_results.csv` contains all evaluated trials.
- [ ] `best_params.yaml` generated and ready for use in B3-B6.
- [ ] `lr_response_curve.png` and `tuning_report.md` generated.

---

## Stop Condition
Stop immediately after creating `best_params.yaml` and generating the report. Do not proceed to B3 automatically.
