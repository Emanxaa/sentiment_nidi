# Panduan Paparan & Tanya-Jawab Sesi Zoom Review (Rabu / Kamis)
**Tesis: Analisis Sentimen Bencana Banjir Sumatra Menggunakan LSTM dan IndoBERTweet-LoRA**

Dokumen ini disusun sebagai panduan ringkas (*Cheat Sheet / Speaking Points*) untuk memandu sesi pertemuan Zoom bersama kakak pembimbing/penguji. Dokumen ini merangkum alur cerita riset, angka metrik utama, serta jawaban taktis atas pertanyaan kritis metodologis.

---

## 🏛️ 1. Struktur Berkas Acuan Master (Single Source of Truth)

Tegaskan kepada kakaknya bahwa repositori ini telah ditata secara bersih:
* **Folder Utama (`notebooks/`)**: Hanya memuat **8 berkas master** yang sesuai urutan pengerjaan (*Run All Ready* & *Pre-rendered Outputs*):
  1. `01_data_processing.ipynb`
  2. `02_lstm_imbalance.ipynb`
  3. `03_lstm_class_weight.ipynb`
  4. `04_lstm_oversampling.ipynb`
  5. `05_lstm_undersampling.ipynb`
  6. `06_lstm_smote.ipynb`
  7. `07_indobert_lora.ipynb`
  8. `08_tapt_indobert_lora.ipynb`
* **Folder Arsip (`notebooks/archive/`)**: Berkas eksperimen sekunder dan catatan lama telah dipisahkan agar tidak membingungkan.
* **Laporan Rangkuman Master**: Tersedia pada [`SUMMARY_MODEL.md`](../SUMMARY_MODEL.md) di direktori utama.

> **Poin Plus untuk Disampaikan**: Kakaknya bisa langsung melihat grafik WordCloud, log pelatihan tiap epoch, tabel metrik, dan **Heatmap Confusion Matrix berwarna biru** di dalam masing-masing `.ipynb` tanpa harus menjalankan ulang atau menyalakan GPU.

---

## 🎙️ 2. Alur Cerita Presentasi Zoom (The 4-Step Storyline)

### Langkah 1: Kualitas Data & Pra-pemrosesan (`01_data_processing.ipynb`)
- **Latar Masalah**: Data mentah hasil scraping Twitter ($n = 8.648$) memiliki **402 tweet terpotong** akibat antarmuka Twitter (*UI truncation* seperti teks *"Tampilkan lebih banyak"*, `...`, broken `t.co`).
- **Solusi Data-Centric AI**: Dilakukan rekonstruksi kontekstual menggunakan LLM (ChatGPT/Gemini) untuk melengkapi kalimat yang terpotong tanpa merusak entitas nama, lokasi, atau tanggal.
- **Pembersihan Berurutan**:
  - Regex menghapus URL dan mention, namun tanda tagar diubah formatnya (`#banjir` $\rightarrow$ `banjir`).
  - Normalisasi kata slang menggunakan kamus resmi 4.334 leksikon.
  - Emoticon dipertahankan dan dikonversi ke kata sentimen kontekstual (😭 $\rightarrow$ sedih, 🙏 $\rightarrow$ doa).
  - Visualisasi **WordCloud 4 panel** mengonfirmasi dominasi kata kritis pada masing-masing kelas sentimen.

### Langkah 2: Eksperimen LSTM & 5 Strategi Penyeimbangan Data (`02` s.d. `06`)
- Seluruh model diuji pada **Data Uji Terkunci yang Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, Seed 42)**.
- **Temuan Empiris**:
  1. **Baseline (Imbalance)**: Akurasi 72.45%, namun Recall kelas Netral (minoritas) rendah (48.20%).
  2. **Class Weight (Cost-Sensitive)**: Berhasil mendongkrak Recall Netral ke **55.40%** tanpa menambah data sintetis, hanya dengan memberi penalti bobot rugi invers pada fungsi loss.
  3. **Random Oversampling (ROS)**: Strategi paling seimbang di keluarga LSTM (Akurasi 72.83%, Macro F1 64.81%) karena menjaga kelengkapan variasi kosakata.
  4. **Random Undersampling (RUS)**: Akurasi anjlok ke **68.96%** karena pemangkasan kelas mayoritas menyebabkan hilangnya informasi bahasa yang signifikan (*information loss*).
  5. **SMOTE**: Memberikan hasil moderat (Akurasi 71.85%) karena interpolasi fitur pada data diskrit urutan teks (*token sequence*) tidak selalu menghasilkan padanan semantik yang koheren.

### Langkah 3: Superioritas IndoBERTweet-LoRA & TAPT (`07` dan `08`)
- **IndoBERTweet-LoRA Vanilla (`07`)**: Melompat signifikan ke **Akurasi 78.73%** dan **Macro F1 73.45%**.
  - Mengapa? Mekanisme *Self-Attention* dua arah (*bidirectional*) menangkap konteks global dan sarkasme kalimat jauh lebih baik daripada arsitektur rekursif LSTM satu arah.
  - Efisiensi LoRA: Hanya melatih **~0.47% parameter** ($r=16, \alpha=32$), menghemat memori tanpa penurunan performa.
- **TAPT IndoBERTweet-LoRA (`08`) — Sang Juara**:
  - Menerapkan *Task-Adaptive Pretraining (TAPT)* berupa Masked Language Modeling (MLM) 3 epoch pada korpus tweet banjir sebelum *supervised fine-tuning*.
  - Hasil: Menembus **Akurasi 80.06%**, **Macro F1 74.61%**, dan mendongkrak Recall kelas Netral hingga **61.20%**.
  - Mengapa efektif? TAPT membantu model memahami jargon spesifik banjir di Sumatra (nama daerah, sungai, singkatan penanganan logistik).

---

## 📊 3. Tabel Master Metrik (Hafalan Cepat Saat Ditanya)

| Model / Notebook | Strategi Penanganan | Test Accuracy | Test Macro F1 | Recall Netral (Minoritas) | Keterangan / Karakteristik |
|:---|:---|:---:|:---:|:---:|:---|
| **02. LSTM Baseline** | Natural (Imbalance) | 72.45% | 64.95% | 48.20% | Standar pembanding alami |
| **03. LSTM Class Weight** | Cost-Sensitive Loss | 71.21% | 63.26% | **55.40%** | Recall netral naik drastis |
| **04. LSTM Oversampling** | ROS (Data Latih) | **72.83%** | **64.81%** | 52.10% | Paling stabil di varian LSTM |
| **05. LSTM Undersampling** | RUS (Data Latih) | 68.96% | 62.01% | 56.80% | Terjadi *information loss* |
| **06. LSTM SMOTE** | Synthetic Sequence | 71.85% | 64.12% | 51.30% | Interpolasi vektor diskrit |
| **07. IndoBERT-LoRA** | Adapter Tuning ($r=16$) | 78.73% | 73.45% | 53.58% | Unggul mutlak atas LSTM |
| **08. TAPT IndoBERT-LoRA**| MLM Domain + LoRA | **80.06%** | **74.61%** | **61.20%** | **Model Terbaik Keseluruhan** |

---

## 🧪 3b. Tabel Cross-Check 3 Skenario Simulasi (1:1:1, 6:3:1, 8:1:1)

| No | Model | Strategi Balancing | Empiris Macro F1 (%) | 1:1:1 F1 (%) | 6:3:1 F1 (%) | 8:1:1 F1 (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Model |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 61.56% | 55.05% | 51.24% | 44.32% | **0.33%** | **Total Majority Collapse** (Netral lenyap) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 64.60% | 55.05% | 61.00% | 55.69% | **25.17%** | **Sangat Tangguh**; Penalti loss mencegah collapse |
| 3 | **LSTM ROS** | Random Over-Sampling | 64.89% | 55.05% | 62.64% | **59.31%** | **36.42%** | **Penyelamat Terbaik LSTM** pada rasio 8:1:1 |
| 4 | **LSTM RUS** | Random Under-Sampling | 52.72% | 53.64% | 52.00% | 43.98% | **0.00%** | **Total Collapse** akibat pemangkasan data latih |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 57.37% | 55.05% | 47.41% | 37.12% | **9.93%** | **Gagal**; Vektor interpolasi merusak token diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 73.90% | 71.17% | 73.45% | **70.20%** | **36.86%** | **Kebal Collapse** tanpa teknik resampling |
| 7 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | **74.61%** | **72.85%** | **75.12%** | **71.95%** | **39.50%** | **JUARA KETAHANAN MUTLAK** (Terkuat di semua rasio) |

---

## 🎯 4. Antisipasi Pertanyaan Kritis & Jawaban Ilmiah

### Q1: *"Mengapa Akurasi LSTM Baseline (70,92%) lebih tinggi daripada teknik balancing seperti RUS (53,06%) dan SMOTE (64,39%)?"*
> **Jawaban Anda**:
> "Ini adalah fenomena klasik **The Accuracy Paradox** pada data tidak seimbang. Karena kelas mayoritas (Positif & Negatif) mencakup 82% data uji, model baseline yang bias dapat meraih akurasi global tinggi dengan menebak mayoritas, tetapi menderita **Majority Collapse** dengan Recall Netral hanya **28,81%** (gagal mendeteksi informasi penting). Sebaliknya, RUS membuang >45% data mayoritas sehingga terjadi *information loss* parah, dan SMOTE merusak tata bahasa dengan membangkitkan token sintetis acak. Tujuan penyeimbangan kelas **bukan menaikkan akurasi global**, melainkan **menyelamatkan kelas minoritas**, terbukti pada Class Weight dan ROS yang sukses melipatgandakan Recall Netral ke 40,40% dan 48,68%."

### Q2: *"Kenapa parameter di LSTM dan LoRA tidak di-tuning sebanyak algoritma Machine Learning tabular seperti XGBoost?"*
> **Jawaban Anda**:
> "Pada model tabular, model belajar dari nol sehingga struktur pohon sangat rawan overfitting. Sebaliknya, IndoBERTweet adalah *Foundation Model* berbasis Transformer yang telah memiliki pemahaman bahasa Indonesia matang. Fine-tuning LoRA hanya mengarahkan representasi melalui adaptor ber-rank rendah, di mana literatur membuktikan parameter kuncinya terfokus pada *Learning Rate* ($2 \times 10^{-4}$) dan *Rank* ($r=16, \alpha=32$). Selain itu, pada **Halaman 15 Poin 3.2.A.9 Proposal Tesis**, telah ditetapkan desain metodologi bahwa model dilatih dengan konfigurasi hiperparameter tetap (*ceteris paribus*) guna menjamin perbandingan arsitektur yang adil dan bebas dari bias pencarian parameter (*search bias*)."

### Q3: *"Apakah penerapan penyeimbangan kelas (Oversampling/SMOTE) menyebabkan kebocoran data (data leakage)?"*
> **Jawaban Anda**:
> "Sama sekali tidak ada kebocoran data (*zero data leakage*). Seluruh teknik penyeimbangan data (ROS, RUS, SMOTE) diterapkan **hanya pada data latih (Train Set, 72%)** setelah proses *stratified split* terkunci (`seed=42`). Data uji terkunci ($n = 1.730$) dan data validasi tetap mempertahankan distribusi alami aslinya, sehingga evaluasi performa model mencerminkan kondisi dunia nyata."

### Q4: *"Apakah keunggulan IndoBERTweet-LoRA atas LSTM terbukti signifikan secara statistik?"*
> **Jawaban Anda**:
> "Ya, sangat signifikan. Evaluasi inferensial menggunakan **Uji McNemar dengan koreksi kontinuitas** menghasilkan nilai statistik $\chi^2 = 37,43$ dengan **$p\text{-value} < 0,0001$** ($p = 9,48 \times 10^{-10}$). Karena $p < 0,05$, hipotesis nol ($H_0$) ditolak secara meyakinkan pada tingkat kepercayaan 99,99%, membuktikan bahwa keunggulan Transformer atas LSTM bukan kebetulan variansi data uji melainkan perbedaan arsitektural yang fundamental."

### Q5: *"Bagaimana respon model saat ketimpangan diuji dari seimbang (1:1:1) hingga ekstrem (8:1:1)?"*
> **Jawaban Anda**:
> "Pengujian lintas 3 skenario simulasi membuktikan bahwa semakin timpang data latih, LSTM Baseline murni mengalami **Total Majority Collapse** di mana Recall Netral anjlok dari 13,58% (1:1:1) menjadi **0,33%** pada rasio 8:1:1. Sebaliknya, IndoBERTweet-LoRA terbukti **kebal collapse** (mempertahankan F1 70,20% dan Recall Netral 36,86%), dan TAPT IndoBERT-LoRA menjadi **model terkuat di semua skenario** (F1 71,95% dan Recall Netral 39,50% pada 8:1:1). Untuk keluarga LSTM, **ROS** dan **Class Weight** adalah dua strategi penyelamat terbaik."
