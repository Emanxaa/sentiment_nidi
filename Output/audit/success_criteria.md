# IndoBERTweet-LoRA Empirical Success Gates & Decision Criteria

**Document Version:** 1.0.0  
**Date:** 2026-09-02  
**Role:** ML Research Engineer  
**Scope:** Quantitative Go / No-Go Decision Thresholds for Model Optimization  

---

## 1. Executive Summary

To prevent confirmation bias and ensure rigorous scientific discipline, this framework defines **four sequential, quantitative decision gates**. An experiment or architectural configuration is accepted only if it satisfies the specific gate condition on held-out validation data.

---

## 2. Decision Gate Definitions

### 🚪 Gate 1: Input Representation Selection Gate
* **Evaluation Split:** Validation Partition ($n=692$).
* **Decision Rule:** Compare `clean_text` vs `processed_text_v2` under identical baseline parameters ($r=16, \alpha=32, \text{dropout}=0.1, lr=2\times 10^{-5}$).
* **Acceptance Condition:**
  $$\text{Selected Input} = \arg\max_{I \in \{\text{clean}, \text{v2}\}} \left( \text{Validation Macro F1}(I) \right)$$
* **Fallback Rule:** If $\Delta \text{Macro F1} < 0.5\text{ pp}$, select `processed_text_v2` to maintain structural parity with the LSTM baseline.

---

### 🚪 Gate 2: LoRA Parameter Optimization Gate
* **Evaluation Split:** Validation Partition ($n=692$).
* **Decision Rule:** Evaluate candidate combinations of $(r, \alpha, \text{dropout}, lr)$ against the default baseline configuration.
* **Acceptance Condition:**
  $$\text{Val Macro F1}_{\text{candidate}} > \text{Val Macro F1}_{\text{baseline}} + 1.0\text{ pp}$$
* **Constraint:** Trainable parameter budget must not exceed **1.5% of total parameters** ($< 1.66\text{M trainable weights}$). If a larger rank (e.g. $r=32$) yields $< 0.5\text{ pp}$ improvement over $r=16$, select the more compact $r=16$ model to honor parameter efficiency principles.

---

### 🚪 Gate 3: Baseline Stability & Reproducibility Gate
* **Evaluation Split:** Test Partition ($n=1,730$), 3 independent seeds (`42`, `123`, `456`).
* **Decision Rule:** Evaluate variance across random weight initializations and mini-batch shuffle orders.
* **Acceptance Conditions:**
  1. Standard deviation of Test Macro F1 must satisfy:
     $$\sigma_{\text{Macro F1}} \le 0.0250 \quad (\le 2.5\text{ pp})$$
  2. Standard deviation of Test Accuracy must satisfy:
     $$\sigma_{\text{Accuracy}} \le 0.0200 \quad (\le 2.0\text{ pp})$$
  3. No single seed may collapse (e.g. Neutral Recall $< 0.30$).

---

### 🚪 Gate 4: Final Cross-Model Superiority Gate (vs LSTM)
* **Evaluation Split:** Test Partition ($n=1,730$, untouched ground truth).
* **Official Benchmark Reference (LSTM Baseline Milestone M3):**
  * **LSTM Test Accuracy:** **72.45%**
  * **LSTM Test Macro F1:** **64.95%**
  * **LSTM Test Precision:** **67.28%**
  * **LSTM Test Recall:** **63.78%**
* **Acceptance Conditions for Thesis Claims:**
  1. **Primary Gate:** Mean IndoBERTweet-LoRA Test Macro F1 must achieve parity or superiority over LSTM:
     $$\overline{\text{Macro F1}}_{\text{IndoBERT}} \ge 64.95\% \quad (\Delta \ge 0.0\text{ pp})$$
  2. **Statistical Significance Gate:** Paired McNemar test on 1,730 test predictions must yield:
     $$p < 0.05$$
  3. All future claims of transformer superiority in the thesis manuscript must reference these exact empirical benchmarks.
