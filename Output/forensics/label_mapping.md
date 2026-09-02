# Milestone B2.5 — Phase 3: Label Mapping Forensic Report

**Generated on:** `2026-09-02`  
**Focus:** Verification of class polarity encoding across preprocessing, tokenizer dictionaries, and classification layers.

---

## 1. Class Taxonomy

| Sentiment Text | Integer Label | HuggingFace `id2label` | HuggingFace `label2id` | Actual Support in Test ($n=1,730$) |
| :--- | :---: | :--- | :--- | :---: |
| **negatif** | `0` | `"negatif"` | `0` | 937 (54.16%) |
| **netral**  | `1` | `"netral"`  | `1` | 302 (17.46%) |
| **positif** | `2` | `"positif"` | `2` | 491 (28.38%) |

---

## 2. Audit Findings

* **Preprocessing Check:** In `legacy_notebooks/01_preprocessing.ipynb` (Cell 17):
  ```python
  label_map = {'negatif': 0, 'netral': 1, 'positif': 2}
  data_new['label'] = data_new['sentimen'].map(label_map).astype(int)
  ```
* **Model Configuration Check:** In `legacy_notebooks/04_model_indobertweet_lora.ipynb` (Cell 10):
  ```python
  id2label = {0: "negatif", 1: "netral", 2: "positif"}
  label2id = {"negatif": 0, "netral": 1, "positif": 2}
  ```
* **Evaluation Code Check:** Both legacy `compute_metrics` and current modular metrics evaluate predictions using `np.argmax(logits, axis=1)` mapped directly to labels `[0, 1, 2]`.
* **Conclusion:** **No label inversion, class rotation, or off-by-one mapping error exists.**
