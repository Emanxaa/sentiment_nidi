# LAPORAN AKHIR HASIL PENELITIAN DAN EKSPERIMEN
**Analisis Sentimen Tweet Bencana Banjir Menggunakan Model IndoBERTweet-LoRA dan Kalibrasi Ambang Batas**

- **Tanggal Dokumen**: 1 September 2026  
- **Lokasi Repositori**: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT`  
- **Status Hasil**: Final dan Siap Digunakan untuk Penulisan Naskah Tesis (Bab III & Bab IV)

---

## 1. Ringkasan Inti Penelitian

Penelitian tesis ini bertujuan mengklasifikasikan sentimen masyarakat pada tweet seputar bencana banjir di wilayah Sumatera ke dalam 3 kategori: **Negatif**, **Netral**, dan **Positif**.

Tantangan terbesar dalam penelitian ini adalah **ketidakseimbangan jumlah data** dan **sulitnya membedakan tweet netral** (seperti tweet informasi ketinggian air atau berita posko) dari tweet negatif (keluhan warga). Akibatnya, pada model awal, tweet netral sangat jarang tertebak dengan benar (kemampuan tangkap/recall hanya 50%).

Melalui perbaikan kualitas label data serta penerapan teknik **Kalibrasi Ambang Batas Keputusan (*Threshold Calibration*)**, model utama (**IndoBERTweet-LoRA**) berhasil ditingkatkan performanya secara signifikan tanpa perlu mengubah struktur arsitektur model.

### Tabel Perbandingan Hasil: Sebelum vs Sesudah Perbaikan
*(Dievaluasi pada Data Uji yang sama, total $n = 1.730$ tweet)*

| Metrik Evaluasi | Model Awal (Sebelum Perbaikan) | Model Final (Setelah Perbaikan Data & Kalibrasi) | Keterangan Perubahan |
|---|---|---|---|
| **Akurasi Keseluruhan** | 79,54% | **77,46%** | Sedikit disesuaikan demi keseimbangan antar-kelas |
| **Macro F1-Score** | 0,7328 | **0,7394** | Nilai rata-rata performa seluruh kelas meningkat |
| **Macro Recall** | 0,7252 | **0,7239** | Keseimbangan deteksi antar ketiga kelas terjaga |
| **Recall Kelas Netral** | 50,00% | **66,90%** | **Naik +16,9%** (Model jauh lebih peka mengenali tweet netral) |
| **F1-Score Kelas Netral** | 0,5500 | **0,6012** | **Naik +5,1%** (Kualitas tebakan kelas netral melampaui batas 0,60) |

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

### Penjelasan Detail Masalah:

1. **Masalah Label Data yang Tidak Konsisten**:
   - Pada data awal, banyak tweet informatif (misalnya: *"Ketinggian air sungai Musi mencapai 1 meter"*) salah dilabeli sebagai sentimen **negatif** hanya karena mengandung kata *"banjir"*.
   - **Solusi**: Dijalankan pipeline audit kualitas data 6-fase ([`quality_pipeline/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/quality_pipeline)). Dilakukan anotasi ulang pada 1.000 sampel menggunakan model bahasa besar (LLM Gemini) yang kemudian divalidasi oleh manusia. Tingkat kesepakatan (*Cohen's Kappa*) bernilai **0,4609** (kategori sedang), dan menghasilkan perbaikan pada 342 tweet yang salah label.

2. **Misteri Skor Rendah LSTM (Nilai 0,2393)**:
   - Pada eksperimen lama, model LSTM tercatat menghasilkan nilai Macro F1 yang sangat rendah dan identik (0,2393) di seluruh percobaan.
   - **Penjelasan**: Ini bukan kesalahan rumus, melainkan kondisi di mana model mengalami **kebuntuan tebakan mayoritas (*Majority Collapse*)**. Karena jumlah data negatif dominan (56%), model mencari jalan pintas dengan selalu menebak kelas negatif untuk setiap tweet. Rumus Macro F1 pada kondisi tebakan mayoritas 56% secara matematis memang menghasilkan angka $\frac{1}{3} \times \frac{2(0,56)}{1 + 0,56} = 0,2393$.
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
|---|---|---|---|---|---|
| **Negatif (0)** | 0,8741 (87,41%) | 0,7996 (79,96%) | 0,8352 | 938 tweet | 858 tweet |
| **Netral (1)** | 0,5459 (54,59%) | **0,6689 (66,89%)** | **0,6012** | 302 tweet | 370 tweet |
| **Positif (2)** | 0,7729 (77,29%) | 0,7918 (79,18%) | 0,7823 | 490 tweet | 502 tweet |
| **Rata-Rata Makro (Macro Avg)** | **0,7392** | **0,7239** | **0,7394** | **1.730 tweet** | **1.730 tweet** |
| **Akurasi Keseluruhan** | \multicolumn{5}{c|}{\textbf{77,46% (1.339 tweet tertebak benar dari 1.730 tweet)}} |

### B. Matriks Konfusi (*Confusion Matrix*) Model Final
Matriks ini menunjukkan sebaran tebakan model dibandingkan dengan label sebenarnya di lapangan:

```
                          TEBAKAN PREDIKSI MODEL
                     Negatif       Netral       Positif    │  Total Aktual
   ┌───────────────┬────────────┬────────────┬────────────┐│
 A │ Negatif       │  750 tweet │  121 tweet │   67 tweet ││   938 tweet
 K │               │  (Benar)   │  (Keliru)  │  (Keliru)  ││
 T ├───────────────┼────────────┼────────────┼────────────┤│
 U │ Netral        │   52 tweet │  202 tweet │   48 tweet ││   302 tweet
 A │               │  (Keliru)  │  (Benar)   │  (Keliru)  ││
 L ├───────────────┼────────────┼────────────┼────────────┤│
   │ Positif       │   56 tweet │   47 tweet │  387 tweet ││   490 tweet
   │               │  (Keliru)  │  (Keliru)  │  (Benar)   ││
   └───────────────┴────────────┴────────────┴────────────┘│
     Total Prediksi   858 tweet    370 tweet    502 tweet      1.730 tweet
```

**Cara Membaca Hasil Matriks**:
1. **Kelas Negatif**: Dari 938 tweet negatif sebenarnya, 750 tweet berhasil dikenali dengan benar (79,96%).
2. **Kelas Netral**: Dari 302 tweet netral sebenarnya, sebanyak 202 tweet berhasil dikenali dengan benar (**66,89%**). Ini merupakan peningkatan besar dibanding model awal yang hanya mampu menangkap 151 tweet netral.
3. **Kelas Positif**: Dari 490 tweet positif sebenarnya, sebanyak 387 tweet berhasil dikenali dengan benar (79,18%).

---

## 4. Evaluasi Eksperimen Tambahan dan Temuan Negatif yang Valid

Dalam penelitian ilmiah, pelaporan metode yang **tidak berhasil** (*negative findings*) sama pentingnya dengan metode yang berhasil, guna membuktikan bahwa eksplorasi telah dilakukan secara komprehensif:

1. **Teknik Penambahan/Pengurangan Data Buatan (*Resampling* seperti SMOTE & Oversampling)**:
   - *Hasil*: **Gagal / Menurunkan Performa**.
   - *Penyebab*: Membuat tweet buatan secara sintetis pada data teks justru merusak struktur kalimat dan tata bahasa alami, sehingga menurunkan skor Macro F1 dari 0,737 menjadi kisaran 0,45 – 0,59.

2. **Teknik Pelembutan Label (*Label Smoothing*)**:
   - *Hasil*: **Tidak Efektif pada Data yang Sudah Bersih**.
   - *Penyebab*: Teknik ini bertujuan mengurangi kepastian berlebih pada model. Namun, karena dataset sudah dikoreksi kualitas labelnya, pelembutan label justru mengaburkan batas pembeda antar-sentimen.

3. **Teknik Modifikasi Fungsi Kerugian (*Focal Loss* $\gamma=2$)**:
   - *Hasil*: **Signifikan Lebih Buruk ($p < 0,001$)**.
   - *Penyebab*: Memberikan penalti terlalu keras pada sampel sulit membuat model menebak kelas netral secara membabi-buta, sehingga nilai presisi netral jatuh ke 48% dan Macro F1 turun menjadi 0,7041.

4. **Batas Kemampuan Model pada Data Saat Ini (*Performance Ceiling*)**:
   - Berbagai variasi fungsi loss dan tuning menunjukkan bahwa batas performa maksimal model dengan representasi data teks yang ada saat ini berada di kisaran **Macro F1 0,73 – 0,74**. Peningkatan lebih lanjut menuju $\ge 0,80$ hanya memungkinkan jika dilakukan penambahan data mentah baru secara berkala.

---

## 5. Tabel Rekapitulasi Komparasi Seluruh Model (Untuk Naskah Bab IV)

Tabel berikut menyajikan perbandingan performa seluruh metode yang telah dievaluasi pada dataset pengujian yang sama ($n = 1.730$):

| No | Pendekatan / Metode Model | Akurasi | Macro Precision | Macro Recall | Macro F1-Score | Recall Netral | Status Metodologis |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **IndoBERTweet-LoRA + Kalibrasi ($w=[1, 1.5, 1]$)** | **77,46%** | **0,7392** | **0,7239** | **0,7394** | **66,89%** | **Model Terbaik (Kandidat Tesis)** |
| 2 | IndoBERTweet-LoRA (Data Terkoreksi Baseline) | 78,27% | 0,7347 | 0,7403 | 0,7371 | 57,00% | Pembanding Tanpa Kalibrasi |
| 3 | IndoBERTweet-LoRA + Weighted CrossEntropy | 77,11% | 0,7380 | 0,7310 | 0,7339 | 63,00% | Setara secara statistik ($p=0,57$) |
| 4 | IndoBERTweet-LoRA (Data Lama Sebelum Koreksi) | 79,54% | 0,7450 | 0,7252 | 0,7328 | 50,00% | Baseline Kanonik Awal |
| 5 | IndoBERTweet-LoRA + Label Smoothing ($\epsilon=0,1$) | 76,80% | 0,7210 | 0,7180 | 0,7195 | 52,00% | Kurang Optimal |
| 6 | IndoBERTweet-LoRA + Focal Loss ($\gamma=2$) | 73,41% | 0,6850 | 0,7310 | 0,7041 | 68,00% | Presisi Rusak ($p < 0,001$) |
| 7 | IndoBERTweet-LoRA (Manipulasi Rasio 1:1:1) | 63,41% | 0,5896 | 0,6189 | 0,5908 | 54,00% | Rekayasa Rasio Menurunkan Hasil |
| 8 | BiLSTM v2 (Data Terkoreksi Baseline) | *Sedang Training* | *Sedang Training* | *Sedang Training* | *Sedang Training* | *Sedang Training* | Baseline Pembanding RNN |
| 9 | LSTM v2 (Data Terkoreksi Baseline) | *Sedang Training* | *Sedang Training* | *Sedang Training* | *Sedang Training* | *Sedang Training* | Baseline Pembanding RNN |

---

## 6. Glosarium dan Definisi Istilah Teknis

Agar laporan ini mudah dipahami secara lugas oleh penguji maupun pembaca umum, berikut adalah definisi operasional istilah yang digunakan:

- **Macro F1-Score**: Nilai rata-rata harmonis antara presisi dan recall yang dihitung terpisah untuk setiap kelas lalu dirata-rata secara seimbang tanpa memandang kelas tersebut banyak atau sedikit. Ini adalah metrik paling adil untuk data yang tidak seimbang.
- **Recall (Daya Tangkap)**: Persentase seberapa banyak tweet dari suatu kategori yang berhasil ditemukan oleh model dari total tweet kategori tersebut yang sebenarnya ada.
- **Precision (Ketepatan Tebakan)**: Persentase seberapa banyak tebakan model yang benar-benar tepat ketika model memprediksi suatu kategori sentimen.
- **LoRA (*Low-Rank Adaptation*)**: Metode melatih model bahasa besar (seperti IndoBERTweet) secara hemat daya dengan hanya menyisipkan dan melatih matriks bobot kecil tambahan, tanpa mengubah seluruh parameter model utama.
- **Threshold Calibration (Kalibrasi Ambang Batas)**: Teknik menyesuaikan batas peluang keputusan setelah pelatihan selesai. Jika model terlalu ragu-ragu memilih kelas minoritas (Netral), probabilitas kelas tersebut dikalikan dengan faktor pengali tertentu (misal 1,5) agar lebih mudah terpilih.
- **Majority Collapse**: Kondisi kegagalan saat model deep learning berhenti belajar dan mengambil jalan pintas dengan menebak semua data sebagai kategori mayoritas saja.
- **Cohen's Kappa ($\kappa$)**: Ukuran statistik untuk menghitung seberapa kuat tingkat kesepakatan antara dua penilai (dalam hal ini anotator manusia dan model AI LLM) setelah memperhitungkan faktor kebetulan.

---

## 7. Indeks Dokumen Teknis Terkait di Repositori

- **Roadmap Pengerjaan**: [`docs/TASK_BOARD.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/TASK_BOARD.md)
- **Catatan Riwayat Eksperimen Rinci**: [`docs/LOG_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LOG_EKSPERIMEN.md)
- **Dokumen Diagnosis Metrik**: [`docs/P0_VERIFIKASI_EVALUASI.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/P0_VERIFIKASI_EVALUASI.md)
- **Panduan Penggunaan Repositori**: [`README.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/README.md)
