# Milestone B2.5 — Phase 6: Metric Audit Forensic Report

**Generated on:** `2026-09-02`  
**Focus:** Verification of Macro F1, Weighted F1, and Precision/Recall calculations.

---

## 1. Metric Definition Audit

1. **Question:** Did the historical experiment report Weighted F1 instead of Macro F1?
2. **Audit Findings:**
   * In `legacy_notebooks/04_model_indobertweet_lora.ipynb` (Cell 20):
     ```python
     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
         y_test,
         y_pred_bert_emp,
         average='macro',
         zero_division=0
     )
     ```
   * Output recorded in `baseline/B03_indobert/hasil_indobertweet_lora_empiris_with_emoticon.csv`:
     * Accuracy: **0.78728** (78.73%)
     * Precision Macro: **0.74235** (74.24%)
     * Recall Macro: **0.72916** (72.92%)
     * Macro F1: **0.73445** (73.45%)
   * We independently calculated the test predictions from `hasil_prediksi_indobertweet_lora_empiris.csv`:
     * Macro F1: **`0.73445` (73.45%)**
     * Weighted F1: **`0.78397` (78.40%)**
3. **Conclusion:** **The historical experiment DID INDEED REPORT MACRO F1 (`average="macro"`).** There was no accidental substitution of Weighted F1.
