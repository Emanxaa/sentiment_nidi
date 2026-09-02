# Data Audit Report - banjir.csv

Generated at: `2026-09-02 03:35:09`  
Source: `Data/raw/banjir.csv`  

---

## 1. Dataset Overview (Part A)

* **Total Rows:** 8,648
* **Total Columns:** 8
* **Duplicate Rows (based on `text`):** 0
* **Rows with at least one candidate flag:** 4,344 (50.23%)

### Column Schema & Missing Values

| Column | Data Type | Missing Count | Missing Percentage |
| :--- | :--- | :--- | :--- |
| `text` | `object` | 0 | 0.00% |
| `clean_text` | `object` | 0 | 0.00% |
| `created_at` | `object` | 0 | 0.00% |
| `keyword` | `object` | 0 | 0.00% |
| `processed_text` | `object` | 1 | 0.01% |
| `sentimen` | `object` | 0 | 0.00% |
| `label` | `int64` | 0 | 0.00% |
| `emoticon` | `object` | 8,483 | 98.09% |

### Label Distribution (`label`)

| Label | Count | Percentage |
| :--- | :--- | :--- |
| `0` | 4,686 | 54.19% |
| `2` | 2,452 | 28.35% |
| `1` | 1,510 | 17.46% |

### Sentiment Distribution (`sentimen`)

| Sentimen | Count | Percentage |
| :--- | :--- | :--- |
| `negatif` | 4,686 | 54.19% |
| `positif` | 2,452 | 28.35% |
| `netral` | 1,510 | 17.46% |

---

## 2. Candidate Detection Summary (Part B & C)

| Flag | Count | Percentage |
| :--- | :--- | :--- |
| `has_truncation` | 402 | 4.65% |
| `has_mention` | 2,262 | 26.16% |
| `has_hashtag` | 2,112 | 24.42% |
| `has_url` | 2 | 0.02% |
| `has_unicode` | 75 | 0.87% |
| `has_engagement` | 154 | 1.78% |
| `has_html` | 3 | 0.03% |

---

## 3. Flag Details & Downstream Mapping

| Flag | Description / Matching Condition | Downstream Task Target |
| :--- | :--- | :--- |
| `has_truncation` | Contains UI truncation text (*Tampilkan lebih banyak*, *View a thread*, *Lihat selengkapnya*) or ending ellipsis `...` / `…` | **Task 02: LLM Completion** |
| `has_mention` | Contains `@username` handles | **Task 03: Regex Refinement** |
| `has_hashtag` | Contains `#hashtag` topics | **Task 03: Regex Refinement** |
| `has_url` | Contains web links (`http://`, `https://`, `t.co/`, etc.) | **Task 03: Regex Refinement** |
| `has_unicode` | Contains unusual Unicode (mojibake `Â`, math bold symbols, non-Latin scripts, braille) | **Task 03 / Preprocessing** |
| `has_engagement` | Contains engagement count artifacts (e.g. `2 rb`, `35 rb`, `1 jt`) | **Task 03: Regex Refinement** |
| `has_html` | Contains HTML entities/tags (`&amp;`, `&ndash;`, `<br>`) | **Task 03: Regex Refinement** |

---

## 4. Integrity Verification

* Original row count preserved: **Yes** (8,648 rows).
* Original text and labels modified: **No** (Audit and candidate detection performed strictly read-only).
* Output dataset saved to: `Data/interim/audit.csv`.
