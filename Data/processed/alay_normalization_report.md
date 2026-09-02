# Kamus Alay Normalization Report - Task 04

Generated at: `2026-09-02 05:01:00`  
Input: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\interim\regex_clean.csv`  
Output: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\processed\banjir_processed_v2.csv`  
Lexicon Dictionary: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\kamus\colloquial-indonesian-lexicon.csv` (4,334 entries)  

---

## 1. Summary Statistics

* **Total Rows:** 8,648
* **Rows Normalized:** 3,316 (38.34%)
* **Unmodified Rows:** 5,332 (61.66%)
* **Lookup Method:** Pure deterministic dictionary replacement (no LLM hallucination).

---

## 2. Before / After Normalization Examples

### Example 1

**Before (`regex_text`):**
```text
Polisi : Sulit menemukan siapa pemilik kayu yg menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH. Presiden, Mentri, Taipan Pemegang HPH, Pengusaha Sawit. Emang gak niat ungkap, krn kelas Paus semua.
```

**After (`processed_text_v2`):**
```text
Polisi : Sulit menemukan siapa pemilik kayu yang menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH. Presiden, Mentri, Taipan Pemegang HPH, Pengusaha Sawit. Memang enggak niat ungkap, karena kelas Paus semua.
```

### Example 2

**Before (`regex_text`):**
```text
Konsistensi Toba Pulp Lestari menerapkan prinsip keberlanjutan dalam operasi usahanya pantas diapresiasi dan didukung, industri lestari seperti TPL adalah masa depan, bencana Sumatera 2025 harus jd pengingat bagi kita semua.. Pakar Unpad: Penguatan Pengelolaan DAS Mendesak untuk Tekan Dampak Banjir di Sumatra.
```

**After (`processed_text_v2`):**
```text
Konsistensi Toba Pulp Lestari menerapkan prinsip keberlanjutan dalam operasi usahanya pantas diapresiasi dan didukung, industri lestari seperti TPL adalah masa depan, bencana Sumatera 2025 harus jadi pengingat bagi kita semua.. Pakar Unpad: Penguatan Pengelolaan DAS Mendesak untuk Tekan Dampak Banjir di Sumatra.
```

### Example 3

**Before (`regex_text`):**
```text
Be honest, what do you see? This isn’t the result of a natural disaster. It’s a landscape teetering on the edge of an ecosystem collapse. We have to prevent this from happening again. Aceh, Sumatra One month after flash floods and landslides in November 2025.
```

**After (`processed_text_v2`):**
```text
Be honest, what di you sih? This isn’t the result of a natural disaster. It’s a landscape teetering on the edge of an ecosystem collapse. We have tapi prevent this from happening again. Aceh, Sumatra One month after flash floods and landslides ini November 2025.
```

### Example 4

**Before (`regex_text`):**
```text
BANJIR susulan masih menjadi kekhawatiran masyarakat di sejumlah wilayah terdampak bencana di Sumatra. Bahkan di penghujung 2025, Selasa (30/12), hujan deras kembali mengguyur Kabupaten Bener Meriah, Aceh. ugm ews meriahbener aceh pkm UGM Pasang EWS Banjir di Bener Meriah
```

**After (`processed_text_v2`):**
```text
BANJIR susulan masih menjadi kekhawatiran masyarakat di sejumlah wilayah terdampak bencana di Sumatra. Bahkan di penghujung 2025, Selasa (30/12), hujan deras kembali mengguyur Kabupaten Benar Meriah, Aceh. ugm ews meriahbener aceh pkm UGM Pasang EWS Banjir di Benar Meriah
```

### Example 5

**Before (`regex_text`):**
```text
Kebenaran tentang banjir Sumatra Siklon Tropis pada Tahun 2025 Sumber: Death toll after tropical cyclones hit Indonesia's Sumatra rises to 442 - Nikkei Asia
```

**After (`processed_text_v2`):**
```text
Kebenaran tentang banjir Sumatra Siklon Tropis pada Tahun 2025 Sumber: Death toll after tropical cyclones hit Indonesia's Sumatra rises tapi 442 - Nikkei Asia
```

---

## 3. Integrity Verification

* **Row Count Preserved:** Yes (8,648 rows).
* **Labels and Sentiments Unchanged:** Yes.
* **Non-lexicon tokens preserved:** Yes.
