# Milestone B1.1 — IndoBERT-LoRA Input Strategy Comparison Report

**Generated on:** `2026-09-02 07:59:36`  
**Evaluation Architecture:** `indolem/indobertweet-base-uncased` + PEFT LoRA ($r=16, \alpha=32, \text{dropout}=0.1$)  
**Seed:** `42`  
**Decision Metric:** Validation Macro F1 (Gate 1)  

---

## 1. Quantitative Input Comparison

| Input Representation | Val Macro F1 | Test Macro F1 | Test Accuracy | Test Precision | Test Recall | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean_text** | **0.2781** | 0.2966 | 0.5647 | 0.4531 | 0.3634 | 86.9s |
| **processed_text_v2** | **0.3028**** (Selected) | 0.3134 | 0.5705 | 0.4476 | 0.3721 | 87.8s |

---

## 2. Decision Gate 1 Resolution

* **Winning Input Representation:** `processed_text_v2`
* **Selection Rationale:** Selected strictly using Validation Macro F1 (0.3028 vs 0.2781).
* **Official Status:** Locked as the primary input text stream for downstream LoRA tuning (B2) and balancing experiments (B3-B6).
