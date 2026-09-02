# Regex Refinement Report - Task 03

Generated at: `2026-09-02 05:00:47`  
Input: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\interim\llm_completed.csv`  
Output: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\interim\regex_clean.csv`  

---

## 1. Summary Statistics

* **Total Rows:** 8,648
* **Rows Modified by Regex Cleaning:** 7,627 (88.19%)
* **Unmodified Rows:** 1,021 (11.81%)
* **Residual URLs:** 0
* **Residual User Mentions (`@username`):** 0
* **Residual Hashtag Symbols (`#word`):** 1 (Words converted to plain text)

---

## 2. Cleaning Rules Applied

1. **Unicode Normalization**: NFKC normalization applied.
2. **URL Removal**: `http://`, `https://`, `www.`, `t.co/`, `bit.ly/` stripped.
3. **Mention Removal**: `@username` stripped.
4. **Engagement Metric Removal**: `2 rb`, `35 rb`, `1 jt`, `ribu`, `juta` stripped.
5. **UI Artifact Removal**: Residual truncation phrases and HTML entities stripped.
6. **Hashtag Preservation**: `#prayforaceh` converted to `prayforaceh`.
7. **Whitespace Normalization**: Whitespace collapsed and trimmed.

---

## 3. Before / After Examples

### Example 1

**Before (`llm_completed_text`):**
```text
Polisi : Sulit menemukan siapa pemilik kayu yg menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH. Presiden, Mentri, Taipan Pemegang HPH, Pengusaha Sawit. Emang gak niat ungkap, krn kelas Paus semua. 72
```

**After (`regex_text`):**
```text
Polisi : Sulit menemukan siapa pemilik kayu yg menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH. Presiden, Mentri, Taipan Pemegang HPH, Pengusaha Sawit. Emang gak niat ungkap, krn kelas Paus semua.
```

### Example 2

**Before (`llm_completed_text`):**
```text
Seorang Ibu harus memanjat tebing demi membawa bayinya ke Puskesmas. Gampong (Desa) Sikundo, Aceh Barat, masih terisolasi. Jalan dan jembatan terputus. Warga terpaksa berpegangan tali di tepi sungai dan tebing untuk melintas. Banjir dan longsor Aceh akibat Siklon 46
```

**After (`regex_text`):**
```text
Seorang Ibu harus memanjat tebing demi membawa bayinya ke Puskesmas. Gampong (Desa) Sikundo, Aceh Barat, masih terisolasi. Jalan dan jembatan terputus. Warga terpaksa berpegangan tali di tepi sungai dan tebing untuk melintas. Banjir dan longsor Aceh akibat Siklon
```

### Example 3

**Before (`llm_completed_text`):**
```text
WASPADA BANJIR PESISIR Durasi: 13 - 20 Januari 2026 2 3 803
```

**After (`regex_text`):**
```text
WASPADA BANJIR PESISIR Durasi: 13 - 20 Januari
```

### Example 4

**Before (`llm_completed_text`):**
```text
BANJIR bandang yang menghancurkan kawasan Aceh, Sumatra Utara, dan Sumatra Barat pada 26-26 November 2025 lalu telah menyisakan kesedihan ratusan ribu korban terdampak 25.905 Rumah di Aceh Timur Rusak akibat Banjir Sumatra, Ribuan Huntara Segera Dibangun 1 221
```

**After (`regex_text`):**
```text
BANJIR bandang yang menghancurkan kawasan Aceh, Sumatra Utara, dan Sumatra Barat pada 26-26 November 2025 lalu telah menyisakan kesedihan ratusan ribu korban terdampak 25.905 Rumah di Aceh Timur Rusak akibat Banjir Sumatra, Ribuan Huntara Segera Dibangun
```

### Example 5

**Before (`llm_completed_text`):**
```text
Be honest, what do you see? This isn’t the result of a natural disaster. It’s a landscape teetering on the edge of an ecosystem collapse. We have to prevent this from happening again. Aceh, Sumatra One month after flash floods and landslides in November 2025. 40
```

**After (`regex_text`):**
```text
Be honest, what do you see? This isn’t the result of a natural disaster. It’s a landscape teetering on the edge of an ecosystem collapse. We have to prevent this from happening again. Aceh, Sumatra One month after flash floods and landslides in November 2025.
```

---

## 4. Integrity Verification

* **Row Count Preserved:** Yes (8,648 rows).
* **Labels and Sentiments Unchanged:** Yes.
* **No Stemming or Slang Normalization Applied:** Yes (Preserved for Task 04).
