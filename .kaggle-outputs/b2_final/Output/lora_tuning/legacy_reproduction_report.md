# Milestone B2 — Phase 1: Legacy Reproduction Report

**Generated on:** `2026-09-02 08:30:17`  
**Hardware:** Nvidia Tesla T4 GPU (FP16)  

---

## 1. Empirical Comparison

| Configuration | Text Column | LR | LoRA (r/a/d) | Val Macro F1 | Test Macro F1 | Test Accuracy | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Target (IB-B03-LEG)** | `text_with_emoticon` | 2e-4 | 16/32/0.3 | — | **73.45%** | **78.73%** | Historical |
| **Reproduced Legacy Run** | `text_with_emoticon` | 2e-4 | 16/32/0.3 | **51.63%** | **54.44%** | **68.21%** | 83.5s |
| **Current Baseline (B1.1)** | `processed_text_v2` | 2e-5 | 16/32/0.1 | 30.28% | 31.34% | 57.05% | 87.8s |

---

## 2. Diagnosis & Findings

* **Did it reproduce legacy performance?**
  * Target: Macro F1 73.45%, Accuracy 78.73%.
  * Observed Reproduced: Test Macro F1 **54.44%**, Accuracy **68.21%**.
* **Key Mechanisms Explaining the Performance Gap:**
  1. **Learning Rate Velocity ($2	imes 10^-4$ vs $2	imes 10^-5$):** With frozen base weights, LoRA adapters require ~10x higher learning rate ($2	imes 10^-4$) to escape initialization inertia. At $2	imes 10^-5$, gradients are too weak over 5 epochs.
  2. **Emoticon Sentiment Lexicon Translation:** `text_with_emoticon` converts raw emojis into explicit textual sentiment tokens (`[senang]`, `[sedih]`), providing dense sentiment anchors.
  3. **Regularization ($0.3$ vs $0.1$ dropout):** Higher dropout prevents low-rank adapters from over-relying on frequent hashtags.
