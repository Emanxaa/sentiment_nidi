# LAPORAN AKHIR HASIL PENELITIAN DAN EKSPERIMEN
**Analisis Sentimen Tweet Bencana Banjir Menggunakan Model IndoBERTweet-LoRA dan Kalibrasi Ambang Batas**

- **Tanggal Dokumen**: 1 September 2026  
- **Lokasi Repositori**: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT`  
- **Dokumen Ringkasan Bukti**: [`que.md`](../que.md)  
- **Status Hasil**: Final dan Terverifikasi untuk Penulisan Naskah Tesis (Bab III & Bab IV)

---

## 1. Ringkasan Inti Penelitian

Penelitian tesis ini bertujuan mengklasifikasikan sentimen masyarakat pada tweet seputar bencana banjir di wilayah Sumatera ke dalam 3 kategori: **Negatif**, **Netral**, dan **Positif**.

Tantangan terbesar dalam penelitian ini adalah **ketidakseimbangan jumlah data (imbalance)** dan **tingginya ambiguitas tweet netral** (seperti tweet informasi ketinggian air sungai atau berita posko) yang sering kali disalahartikan sebagai tweet negatif keluhan warga. Akibatnya, pada model awal, tweet netral sangat jarang tertebak dengan benar (daya tangkap/recall hanya 50%).

Melalui kombinasi perbaikan kualitas anotasi data (*Data-Centric AI*) serta penerapan teknik **Kalibrasi Ambang Batas Keputusan (*Threshold Calibration*)**, model utama (**IndoBERTweet-LoRA**) berhasil ditingkatkan performanya secara signifikan tanpa perlu mengubah arsitektur model.

### Tabel Perbandingan Hasil: Sebelum vs Sesudah Perbaikan
*(Dievaluasi pada Data Uji yang sama, total $n = 1.730$ tweet)*

| Metrik Evaluasi | Model Awal (Sebelum Perbaikan) | Model Final (Setelah Perbaikan Data & Kalibrasi) | Keterangan Perubahan |
|---|:---:|:---:|---|
| **Akurasi Keseluruhan** | 79,54% | **77,46%** | Sedikit disesuaikan demi keseimbangan antar-kelas |
| **Macro F1-Score** | 0,7328 | **0,7394** | Nilai rata-rata performa seluruh kelas meningkat |
| **Macro Recall** | 0,7252 | **0,7532** | Keseimbangan deteksi antar ketiga kelas meningkat |
| **Recall Kelas Netral** | 50,00% | **66,89%** | **Naik +16,9%** (Model jauh lebih peka mengenali tweet netral) |
| **F1-Score Kelas Netral** | 0,5500 | **0,6012** | **Naik +5,1%** (Kualitas tebakan netral melampaui batas $\ge 0,60$) |

---

## 2. Permasalahan Utama yang Ditemukan dan Diselesaikan

Sebelum mencapai hasil akhir, dilakukan audit menyeluruh terhadap kendala-kendala yang sempat terjadi pada eksperimen terdahulu:

```
                                PERJALANAN PERBAIKAN SISTEM
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                        │
  │  [MASALAH 1: Kualitas Label Data]      ──► [SOLUSI 1: Koreksi Anotasi 6-Fase]          │
  │  Banyak tweet berita netral salah          Anotasi ulang 1.000 sampel via LLM & review │
  │  diberi label negatif karena kata          manusia. Baseline F1 naik dari 0,721 ─► 0,737│
  │  "banjir".                                                                             │
  │                                                                                        │
  │  [MASALAH 2: Isu Metrik LSTM 0.2393]   ──► [SOLUSI 2: Perbaikan Pelatihan LSTM]        │
  │  Bukan salah rumus metrik, tapi            Pencegahan penghentian dini (patience=7)     │
  │  model LSTM menebak 100% negatif           agar model belajar tuntas (bebas collapse). │
  │  (Majority Collapse).                                                                  │
  │                                                                                        │
  │  [MASALAH 3: Inkonsistensi Teknis]     ──► [SOLUSI 3: Standardisasi Lingkungan]        │
  │  Multi-GPU Kaggle menggandakan batch       Paksa 1 GPU (CUDA_VISIBLE_DEVICES=0) dan     │
  │  & versi library Transformers berubah.     kunci library Transformers ke versi 4.46.3.  │
  │                                                                                        │
  │  [MASALAH 4: Bottleneck Kelas Netral]  ──► [SOLUSI 4: Kalibrasi Ambang Batas]          │
  │  Peluang prediksi kelas netral             Bobot keputusan w=[1.0, 1.5, 1.0]           │
  │  selalu kalah tipis dari negatif.          Recall Netral melonjak dari 50% ─► 66,9%.   │
  │                                                                                        │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

### Penjelasan Detail Masalah & Solusi:

1. **Masalah Label Data yang Tidak Konsisten**:
   - Pada data awal, banyak tweet informatif (misalnya: *"Ketinggian air sungai Musi mencapai 1 meter"*) salah dilabeli sebagai sentimen **negatif** hanya karena mengandung kata *"banjir"*.
   - **Solusi**: Dijalankan pipeline audit kualitas data 6-fase ([`quality_pipeline/`](../quality_pipeline)). Dilakukan anotasi ulang pada 1.000 sampel menggunakan LLM Gemini yang divalidasi manusia. Tingkat kesepakatan (*Cohen's Kappa*) bernilai **0,4609** (kategori sedang), dan menghasilkan perbaikan pada 342 tweet yang salah label.

2. **Misteri Skor Rendah LSTM (Nilai 0,2393)**:
   - Pada eksperimen lama, model LSTM tercatat menghasilkan nilai Macro F1 yang sangat rendah dan identik (0,2393) di seluruh percobaan.
   - **Penjelasan**: Ini bukan kesalahan rumus metrik, melainkan kondisi di mana model mengalami **kebuntuan tebakan mayoritas (*Majority Collapse*)**. Karena jumlah data negatif dominan (56%), model mencari jalan pintas dengan selalu menebak kelas negatif. Rumus Macro F1 pada kondisi tebakan mayoritas 56% secara matematis memang bernilai $\frac{1}{3} \times \frac{2(0,56)}{1 + 0,56} = 0,2393$.
   - **Solusi**: Parameter pelatihan LSTM diperbaiki dengan menambah kesabaran penghentian dini (*early stopping patience* dari 3 menjadi 7 epoch) serta penurunan laju belajar bertahap (*learning rate reduction*).

3. **Pemisahan Jalur Teks Berdasarkan Karakteristik Model**:
   - Model RNN (LSTM & BiLSTM) membutuhkan kata dasar hasil *stemming* (mengubah kata berimbuhan menjadi kata dasar).
   - Sebaliknya, model Transformer (IndoBERTweet) membutuhkan susunan kalimat asli yang utuh tanpa *stemming* agar mekanisme *self-attention* dapat memahami konteks kalimat secara alami.
   - **Solusi**: Dibuat pemisahan kolom input yang tegas: **`clean_text_lstm`** untuk LSTM/BiLSTM dan **`text_bert`** untuk IndoBERTweet.

4. **Kelemahan Model dalam Mendeteksi Kelas Netral**:
   - Model standar menggunakan aturan pengambilan keputusan probabilitas terbesar ($\text{argmax}$). Karena data negatif jauh lebih banyak, skor probabilitas kelas netral sering kali kalah tipis (misal: probabilitas negatif 0,42 vs netral 0,38), sehingga tweet netral sering terabaikan.
   - **Solusi**: Menerapkan **Kalibrasi Ambang Batas Keputusan (*Threshold Calibration*)** dengan mengalikan probabilitas kelas netral sebesar 1,5 kali sebelum keputusan akhir diambil.

---

## 3. Hasil Akhir Model Terbaik yang Dikunci

Model akhir yang ditetapkan sebagai kandidat utama tesis adalah:
- **Model Dasar**: IndoBERTweet-base (`indolem/indobertweet-base-uncased`)
- **Metode Pelatihan**: Fine-tuning efisien parameter (*Low-Rank Adaptation* / LoRA) dengan konfigurasi rank $r=16$ dan $\alpha=32$.
- **Data Input**: Kolom teks kanonik `text_bert` pada dataset yang sudah dikoreksi labelnya (*Label Corrected*).
- **Aturan Keputusan**: Menggunakan bobot kalibrasi $w = [1,0 \text{ (Negatif)}, 1,5 \text{ (Netral)}, 1,0 \text{ (Positif)}]$.

### A. Tabel Evaluasi Lengkap per Kategori Sentimen (Data Uji, $n = 1.730$)

| Kategori Sentimen | Presisi (*Precision*) | Daya Tangkap (*Recall*) | F1-Score | Jumlah Data Aktual | Jumlah Tebakan Model |
|---|:---:|:---:|:---:|:---:|:---:|
| **Negatif (0)** | 0,8741 (87,41%) | 0,7996 (79,96%) | **0,8352** | 937 tweet | 858 tweet |
| **Netral (1)** | 0,5459 (54,59%) | **0,6689 (66,89%)** | **0,6012** | 302 tweet | 370 tweet |
| **Positif (2)** | 0,7729 (77,29%) | 0,7918 (79,18%) | **0,7823** | 491 tweet | 502 tweet |
| **Rata-Rata Makro (Macro Avg)** | **0,7310** | **0,7532** | **0,7394** | **1.730 tweet** | **1.730 tweet** |
| **Akurasi Keseluruhan** | \multicolumn{5}{c|}{\textbf{77,46% (1.339 tweet tertebak benar dari 1.730 tweet)}} |

### B. Matriks Konfusi (*Confusion Matrix*) Model Final
Matriks ini menunjukkan sebaran tebakan model dibandingkan dengan label sebenarnya di lapangan:

```
                          TEBAKAN PREDIKSI MODEL
                     Negatif       Netral       Positif    │  Total Aktual
   ┌───────────────┬────────────┬────────────┬────────────┐│
 A │ Negatif       │  750 tweet │  121 tweet │   66 tweet ││   937 tweet
 K │               │  (Benar)   │  (Keliru)  │  (Keliru)  ││
 T ├───────────────┼────────────┼────────────┼────────────┤│
 U │ Netral        │   52 tweet │  202 tweet │   48 tweet ││   302 tweet
 A │               │  (Keliru)  │  (Benar)   │  (Keliru)  ││
 L ├───────────────┼────────────┼────────────┼────────────┤│
   │ Positif       │   56 tweet │   47 tweet │  388 tweet ││   491 tweet
   │               │  (Keliru)  │  (Keliru)  │  (Benar)   ││
   └───────────────┴────────────┴────────────┴────────────┘│
     Total Prediksi   858 tweet    370 tweet    502 tweet      1.730 tweet
```

**Cara Membaca Hasil Matriks**:
1. **Kelas Negatif**: Dari 937 tweet negatif sebenarnya, 750 tweet berhasil dikenali dengan benar (79,96%).
2. **Kelas Netral**: Dari 302 tweet netral sebenarnya, sebanyak 202 tweet berhasil dikenali dengan benar (**66,89%**). Ini merupakan peningkatan besar dibanding model awal yang hanya mampu menangkap 151 tweet netral.
3. **Kelas Positif**: Dari 491 tweet positif sebenarnya, sebanyak 388 tweet berhasil dikenali dengan benar (79,18%).

---

## 4. Evaluasi Eksperimen Tambahan dan Temuan Negatif yang Valid

Dalam penelitian ilmiah, pelaporan metode yang **tidak berhasil** (*negative findings*) sama pentingnya dengan metode yang berhasil, guna membuktikan bahwa eksplorasi telah dilakukan secara komprehensif:

1. **Teknik Penambahan Data Buatan (*Resampling* seperti SMOTE & Oversampling)**:
   - *Hasil*: **Gagal / Menurunkan Performa**.
   - *Penyebab*: Membuat tweet buatan secara sintetis pada data teks justru merusak struktur kalimat dan tata bahasa alami, sehingga menurunkan skor Macro F1 dari 0,737 menjadi kisaran 0,45 – 0,59.

2. **Teknik Pelembutan Label (*Label Smoothing $\epsilon=0,1$*)**:
   - *Hasil*: **Tidak Efektif pada Data yang Sudah Bersih** (Val F1 0,6949 vs 0,7099).
   - *Penyebab*: Teknik ini bertujuan mengurangi kepastian berlebih pada model. Namun, karena dataset sudah dikoreksi kualitas labelnya, pelembutan label justru mengaburkan batas pembeda antar-sentimen.

3. **Teknik Modifikasi Fungsi Kerugian (*Focal Loss $\gamma=2$*)**:
   - *Hasil*: **Signifikan Lebih Buruk ($p < 0,0001$)**.
   - *Penyebab*: Memberikan penalti terlalu keras pada sampel sulit membuat model menebak kelas netral secara berlebihan (426 tebakan vs 302 aktual), sehingga presisi netral jatuh ke 48% dan Macro F1 anjlok ke 0,7041.

4. **Batas Kemampuan Model pada Data Saat Ini (*Performance Ceiling*)**:
   - Berbagai variasi fungsi loss dan tuning menunjukkan bahwa batas performa maksimal model dengan representasi data teks yang ada saat ini berada di kisaran **Macro F1 0,73 – 0,74**. Peningkatan lebih lanjut menuju $\ge 0,80$ hanya memungkinkan jika dilakukan adaptasi domain representasi bahasa (seperti TAPT/MLM).

---

## 5. Tabel Rekapitulasi Komparasi Seluruh Model (Untuk Naskah Bab IV)

Tabel berikut menyajikan perbandingan performa seluruh metode baseline dan model transformer yang telah dievaluasi pada dataset pengujian yang sama ($n = 1.730$):

### A. Tabel Master Baseline & Model Comparison

| No | Arsitektur Model | Representasi Fitur / Strategi | Akurasi | Macro Precision | Macro Recall | Macro F1 | Recall Netral | F1 Netral | Cohen's Kappa ($\kappa$) | Status Metodologis |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **IndoBERTweet-LoRA + TAPT + Kalibrasi ($w=[1, 1.4, 1]$)** | Transformer + MLM + Kalibrasi | **79,60%** | **0,7494** | **0,7464** | **0,7479** | **58,94%** | **0,6082** | **0,6492** | 🥇 **Model Terbaik Keseluruhan (P2 TAPT)** |
| 2 | IndoBERTweet-LoRA + TAPT via MLM (P2 Argmax) | Transformer + MLM (3 Epoch) | 80,06% | 0,7583 | 0,7383 | 0,7461 | 52,65% | 0,5894 | 0,6541 | Akurasi Tertinggi ($p=0,0041$) |
| 3 | **IndoBERTweet-LoRA + Kalibrasi ($w=[1, 1.5, 1]$)** | Transformer + LoRA + Kalibrasi | 77,46% | 0,7310 | 0,7532 | **0,7394** | **66,89%** | **0,6012** | **0,6274** | 🥈 **Baseline Terkunci + Kalibrasi** |
| 4 | IndoBERTweet-LoRA (Empiris / B03) | Transformer + LoRA ($r=16, \alpha=32$) | 78,73% | 0,7424 | 0,7292 | 0,7345 | 53,58% | 0,5638 | **0,6387** | Baseline Transformer Standar |
| 5 | IndoBERTweet-LoRA + Class Weight (B03) | Transformer + Weighted Loss | 74,10% | 0,7027 | 0,7375 | 0,7114 | 63,58% | 0,5512 | 0,5788 | Bobot Penalti Kelas |
| 6 | **BiLSTM (Empiris / B02)** | Word Embedding + Bidirectional LSTM | 75,26% | 0,6963 | 0,6837 | **0,6880** | 49,15% | 0,5126 | **0,5782** | Baseline RNN Dua Arah |
| 7 | **LSTM (Baseline / B01)** | Word Embedding + LSTM | 72,66% | 0,6812 | 0,7084 | **0,6899** | 55,12% | 0,5283 | **0,5694** | Baseline RNN Satu Arah |
| 8 | **Linear SVM (LinearSVC)** | TF-IDF (Unigram + Bigram) | 75,78% | 0,7071 | 0,6798 | **0,6878** | 40,40% | 0,4485 | **0,5821** | Baseline Linear Klasik Terbaik |
| 9 | Logistic Regression (L2) | TF-IDF (Unigram + Bigram) | 76,30% | 0,7461 | 0,6611 | 0,6761 | 31,13% | 0,3806 | 0,5768 | Baseline Probabilistik Klasik |
| 10 | SGD Classifier (Log Loss) | TF-IDF (Unigram + Bigram) | 75,95% | 0,7452 | 0,6513 | 0,6661 | 28,48% | 0,3539 | 0,5691 | Baseline Optimasi Gradien |
| 11 | Multinomial Naive Bayes | TF-IDF (Unigram + Bigram) | 73,18% | 0,7382 | 0,6176 | 0,6274 | 22,52% | 0,2976 | 0,5234 | Baseline Probabilitas Bersyarat |
| 12 | Random Forest (100 Trees) | TF-IDF (Unigram + Bigram) | 74,45% | 0,7604 | 0,6145 | 0,6223 | 18,54% | 0,2606 | 0,5392 | Baseline Ensemble Pohon Keputusan |
| 13 | Gradient Boosting | TF-IDF (Unigram + Bigram) | 73,53% | 0,7122 | 0,6108 | 0,6211 | 20,86% | 0,2882 | 0,5285 | Baseline Boosting Sekuensial |

---

### B. Uji Signifikansi Statistik Inferensial (McNemar's Chi-Square Test)

Untuk membuktikan secara ilmiah bahwa keunggulan model Transformer bukan semata faktor kebetulan (*chance*), dilakukan pengujian statistik inferensial McNemar dengan derajat kebebasan $df=1$ dan taraf signifikansi $\alpha = 0,05$:

1. **IndoBERTweet-LoRA vs Baseline LSTM**:
   - $\chi^2 = 38.42, \quad p = 5.71 \times 10^{-10} \quad (p < 0.0001)$
   - **Kesimpulan**: IndoBERTweet-LoRA **unggul signifikan secara statistik** atas LSTM.
2. **IndoBERTweet-LoRA vs Linear SVM (TF-IDF)**:
   - $\chi^2 = 46.18, \quad p = 1.08 \times 10^{-11} \quad (p < 0.0001)$
   - **Kesimpulan**: IndoBERTweet-LoRA **unggul signifikan secara statistik** atas Linear SVM.
3. **BiLSTM vs LSTM**:
   - $\chi^2 = 3.12, \quad p = 0.0773 \quad (p > 0.05)$
   - **Kesimpulan**: Perbedaan performa antara BiLSTM dan LSTM **tidak berbeda signifikan secara statistik**, membuktikan keterbatasan embedding statis pada pemodelan sentimen informal.
4. **IndoBERTweet-LoRA (Argmax) vs IndoBERTweet-LoRA (Kalibrasi $w=[1, 1.5, 1]$)**:
   - $\chi^2 = 0.89, \quad p = 0.3455 \quad (p > 0.05)$
   - **Kesimpulan**: Kalibrasi ambang batas berhasil menaikkan Recall Netral secara drastis (+13.31%) **tanpa menurunkan akurasi umum secara signifikan**.

---

### C. Eksperimen Komprehensif Penyeimbangan Kelas pada LSTM (Milestones M1 — M8)

Untuk menganalisis secara mendalam perilaku arsitektur LSTM terhadap ketimpangan kelas, dievaluasi 5 strategi pada data empiris alami ($n=1.730$) dan 3 skenario simulasi ketimpangan (Balanced 1:1:1, Moderate 6:3:1, Severe 8:1:1) melintasi 3 random seed independen (`42`, `123`, `456`):

#### 1. Performa Empiris pada Distribusi Alami Tweet Banjir (Mean ± SD)
| Strategi Penyeimbangan | Akurasi | Macro F1 | Macro Precision | Macro Recall | Recall Netral | Peringkat Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (M3)** | **72,45% ± 1,07%** | **64,95% ± 2,13%** | **67,28% ± 2,09%** | 63,78% ± 2,12% | 46,25% | **#1 (Terbaik)** |
| **Random Oversampling / ROS (M5)** | 67,05% ± 2,31% | 62,71% ± 1,16% | 63,25% ± 1,32% | 63,73% ± 0.45% | 51,10% | **#2** |
| **Class Weight (M4)** | 65,92% ± 3,60% | 62,70% ± 2,01% | 63,12% ± 1,19% | **65,15% ± 0,13%** | **59,16%** | **#3 (Recall Netral Terbaik)** |
| **Random Undersampling / RUS (M6)** | 62,95% ± 3,60% | 58,92% ± 2,33% | 59,45% ± 1,71% | 60,30% ± 2,10% | 48,01% | **#4** |
| **SMOTE (M7)** | 42,72% ± 1,76% | 40.76% ± 3.07% | 44,91% ± 2,90% | 44,45% ± 1,81% | 38,20% | **#5** |

#### 2. Ketahanan Model pada Skenario Ketimpangan Buatan (Macro F1)
| Strategi | Skenario A (1:1:1) | Skenario B (6:3:1) | Skenario C (8:1:1) | Pola Ketahanan |
|:---|:---:|:---:|:---:|---|
| **Baseline** | **58,16%** | 57,16% | 45,97% | Ambruk drastis (-12.19 pp) pada ketimpangan ekstrem |
| **Class Weight** | **58,16%** | 59,75% | 57,00% | Sangat tangguh mempertahankan F1 pada 8:1:1 (+11,03 pp vs Base) |
| **ROS** | 56,19% | **59,95%** | **58,51%** | **Performa tertinggi pada ketimpangan ekstrem (+12,54 pp vs Base)** |
| **RUS** | 56,19% | 55,81% | 49,77% | Penurunan performa akibat hilangnya leksikon mayoritas |
| **SMOTE** | **58,16%** | 35,74% | 35,12% | Gagal mempertahankan representasi tata bahasa sekuensial |

*Seluruh tabel rekapitulasi numerik dan grafik beresolusi tinggi (300 DPI) tersimpan di direktori terpusat [`Output/summary/`](../Output/summary).*

---

## 6. Glosarium dan Definisi Istilah Teknis

- **Macro F1-Score**: Nilai rata-rata harmonis antara presisi dan recall yang dihitung terpisah untuk setiap kelas lalu dirata-rata secara seimbang tanpa memandang kelas tersebut mayoritas atau minoritas. Metrik paling adil untuk data yang tidak seimbang.
- **Recall (Daya Tangkap)**: Persentase seberapa banyak tweet dari suatu kategori yang berhasil ditemukan oleh model dari total tweet kategori tersebut yang sebenarnya ada.
- **Precision (Ketepatan Tebakan)**: Persentase seberapa banyak tebakan model yang benar-benar tepat ketika model memprediksi suatu kategori sentimen.
- **LoRA (*Low-Rank Adaptation*)**: Metode melatih model bahasa besar secara hemat daya dengan hanya menyisipkan dan melatih matriks bobot kecil tambahan, tanpa mengubah seluruh parameter model utama.
- **Threshold Calibration (Kalibrasi Ambang Batas)**: Teknik menyesuaikan batas peluang keputusan setelah pelatihan selesai. Jika model terlalu ragu-ragu memilih kelas minoritas (Netral), probabilitas kelas tersebut dikalikan dengan faktor pengali tertentu (misal 1,5) agar lebih mudah terpilih.
- **Majority Collapse**: Kondisi kegagalan saat model deep learning berhenti belajar dan mengambil jalan pintas dengan menebak semua data sebagai kategori mayoritas saja.
- **Cohen's Kappa ($\kappa$)**: Ukuran statistik untuk menghitung seberapa kuat tingkat kesepakatan antara dua penilai setelah memperhitungkan faktor kebetulan.

---

## 7. Indeks Dokumen Terkait di Repositori

- **Ringkasan Bukti Proyek 9 Fase**: [`que.md`](../que.md)
- **Roadmap Pengerjaan**: [`docs/TASK_BOARD.md`](TASK_BOARD.md)
- **Catatan Riwayat Eksperimen Rinci**: [`docs/LOG_EKSPERIMEN.md`](LOG_EKSPERIMEN.md)
- **Dokumen Diagnosis Metrik**: [`docs/P0_VERIFIKASI_EVALUASI.md`](P0_VERIFIKASI_EVALUASI.md)
- **Panduan Repositori**: [`README.md`](../README.md)
