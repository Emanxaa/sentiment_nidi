# Milestone B2.5 — Phase 1: Dataset Integrity Forensic Report

**Generated on:** `2026-09-02`  
**Focus:** Cryptographic & structural comparison between raw, legacy, and current dataset artifacts.

---

## 1. Cryptographic Hashes & File Attributes

| Artifact | File Size | SHA256 Hash (Truncated) | Rows | Total Nulls | Text Duplicates |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **`Data/data_banjir.csv` (Raw)** | 5,576,205 B | `38b0734509fa0796f4abdb8897a94d1c...` | 8,648 | 8484 | 0 |
| **`kaggle_dataset/data_preprocessed_with_emoticon.csv` (Legacy)** | 12,813,494 B | `b6138edddd490abd2768ea8ce3603924...` | 8,648 | 8484 | 0 |
| **`Data/processed/banjir_processed_v2.csv` (Current)** | 11,690,139 B | `f0493512ac99b8cd7a2cb6b6ed5e69de...` | 8,648 | 8484 | 0 |

Full SHA256 Checksums:
* `Data/data_banjir.csv`: `38b0734509fa0796f4abdb8897a94d1cdb1388e99bb4f73d1e282d10e422f2cf`
* `kaggle_dataset/data_preprocessed_with_emoticon.csv`: `b6138edddd490abd2768ea8ce3603924442987de695cc6d13425eb30bd161e2d`
* `Data/processed/banjir_processed_v2.csv`: `f0493512ac99b8cd7a2cb6b6ed5e69deff4df3feaf0d6eeae4637410088fbebe`

---

## 2. Row Alignment & Distribution Invariance

* **Row Count Invariance:** All three datasets contain precisely **8,648 rows**.
* **Label Distribution Invariance:**
  * `negatif` (0): **4,685** (54.17%)
  * `positif` (2): **2,453** (28.37%)
  * `netral` (1): **1,510** (17.46%)
* **Row-by-Row Label Concordance:** The `label` column matches **100.0% row-for-row** across all artifacts (`(df_legacy['label'] == df_current['label']).all() == True`).
* **Text Alignment:** The raw `text` column matches **100.0% row-for-row** across all artifacts.

---

## 3. Explaining Hash Divergence

The SHA256 hashes diverge exclusively due to added columns across pipeline evolutions:
1. `data_preprocessed_with_emoticon.csv` (12 columns) contains `text_with_emoticon` and `text_bert`.
2. `banjir_processed_v2.csv` (19 columns) introduces metadata flags (`has_truncation`, `has_mention`, `has_url`), LLM completion outputs (`llm_completed_text`), and the standardized `processed_text_v2`.
