# Preprocessing & Emoticon Handling Report - Task 05

Generated at: `2026-09-02 05:07:13`  
Input: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\processed\banjir_processed_v2.csv`  
Output: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT\Data\processed\data_preprocessed_v2.csv`  

---

## 1. Summary Statistics

* **Total Rows:** 8,648
* **Active Emoji Mappings:** 39 emojis
* **Active Stopwords Filtered:** 121 words
* **Sentiment Keep Words Preserved:** 23 words
* **Label Distribution:**
label
0    4686
2    2452
1    1510

---

## 2. Representation Streams

1. **`text_bert`**: Context-preserved, punctuation-retained, emoji-converted representation for Transformer models (IndoBERTweet-LoRA).
2. **`clean_text_lstm`**: Noise-filtered, sentiment-preserving stopword filtered representation for RNN models (LSTM & BiLSTM).

---

## 3. Transformation Samples

### Example 1

**Input (`processed_text_v2`):**
```text
Polisi : Sulit menemukan siapa pemilik kayu yang menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH. Presiden, Mentri, Taipan Pemegang HPH, Pengusaha Sawit. Memang enggak niat ungkap, karena kelas Paus semua.
```

**IndoBERT (`text_bert`):**
```text
polisi sulit menemukan siapa pemilik kayu yang menyebabkan banjir sumatra akhir 2025. rakyat polisi goblok, percuma sekolah, itu di bpn kan ada siapa pemilik lahan hph. presiden, mentri, taipan pemegang hph, pengusaha sawit. memang enggak niat ungkap, karena kelas paus semua.
```

**LSTM/BiLSTM (`clean_text_lstm`):**
```text
polisi sulit menemukan siapa pemilik kayu menyebabkan banjir sumatra akhir rakyat polisi goblok percuma sekolah bpn kan siapa pemilik lahan hph presiden mentri taipan pemegang hph pengusaha sawit memang enggak niat ungkap kelas paus semua
```

### Example 2

**Input (`processed_text_v2`):**
```text
Seorang Ibu harus memanjat tebing demi membawa bayinya ke Puskesmas. Gampong (Desa) Sikundo, Aceh Barat, masih terisolasi. Jalan dan jembatan terputus. Warga terpaksa berpegangan tali di tepi sungai dan tebing untuk melintas. Banjir dan longsor Aceh akibat Siklon
```

**IndoBERT (`text_bert`):**
```text
seorang ibu harus memanjat tebing demi membawa bayinya ke puskesmas. gampong desa sikundo, aceh barat, masih terisolasi. jalan dan jembatan terputus. warga terpaksa berpegangan tali di tepi sungai dan tebing untuk melintas. banjir dan longsor aceh akibat siklon
```

**LSTM/BiLSTM (`clean_text_lstm`):**
```text
seorang ibu memanjat tebing membawa bayinya puskesmas gampong desa sikundo aceh barat terisolasi jalan jembatan terputus warga terpaksa berpegangan tali tepi sungai tebing melintas banjir longsor aceh akibat siklon
```

### Example 3

**Input (`processed_text_v2`):**
```text
WASPADA BANJIR PESISIR Durasi: 13 - 20 Januari
```

**IndoBERT (`text_bert`):**
```text
waspada banjir pesisir durasi 13 - 20 januari
```

**LSTM/BiLSTM (`clean_text_lstm`):**
```text
waspada banjir pesisir durasi januari
```

### Example 4

**Input (`processed_text_v2`):**
```text
Konsistensi Toba Pulp Lestari menerapkan prinsip keberlanjutan dalam operasi usahanya pantas diapresiasi dan didukung, industri lestari seperti TPL adalah masa depan, bencana Sumatera 2025 harus jadi pengingat bagi kita semua.. Pakar Unpad: Penguatan Pengelolaan DAS Mendesak untuk Tekan Dampak Banjir di Sumatra.
```

**IndoBERT (`text_bert`):**
```text
konsistensi toba pulp lestari menerapkan prinsip keberlanjutan dalam operasi usahanya pantas diapresiasi dan didukung, industri lestari seperti tpl adalah masa depan, bencana sumatera 2025 harus jadi pengingat bagi kita semua.. pakar unpad penguatan pengelolaan das mendesak untuk tekan dampak banjir di sumatra.
```

**LSTM/BiLSTM (`clean_text_lstm`):**
```text
konsistensi toba pulp lestari menerapkan prinsip keberlanjutan operasi usahanya pantas diapresiasi didukung industri lestari tpl masa depan bencana sumatera jadi pengingat semua pakar unpad penguatan pengelolaan das mendesak tekan dampak banjir sumatra
```

### Example 5

**Input (`processed_text_v2`):**
```text
BANJIR bandang yang menghancurkan kawasan Aceh, Sumatra Utara, dan Sumatra Barat pada 26-26 November 2025 lalu telah menyisakan kesedihan ratusan ribu korban terdampak 25.905 Rumah di Aceh Timur Rusak akibat Banjir Sumatra, Ribuan Huntara Segera Dibangun
```

**IndoBERT (`text_bert`):**
```text
banjir bandang yang menghancurkan kawasan aceh, sumatra utara, dan sumatra barat pada 26-26 november 2025 lalu telah menyisakan kesedihan ratusan ribu korban terdampak 25.905 rumah di aceh timur rusak akibat banjir sumatra, ribuan huntara segera dibangun
```

**LSTM/BiLSTM (`clean_text_lstm`):**
```text
banjir bandang menghancurkan kawasan aceh sumatra utara sumatra barat november lalu menyisakan kesedihan ratusan ribu korban terdampak rumah aceh timur rusak akibat banjir sumatra ribuan huntara segera dibangun
```

---

## 4. Integrity Verification

* **Row Count Preserved:** Yes (8,648 rows).
* **Labels and Sentiments Aligned:** Yes (`negatif: 0`, `netral: 1`, `positif: 2`).
