# 🎤 Panduan Paparan & Tanya-Jawab Sesi Zoom Review Tesis
**Judul Tesis**: Analisis Sentimen Bencana Banjir Sumatra Menggunakan LSTM dan IndoBERTweet-LoRA  
**Penulis**: Emanuel M.  

Dokumen ini disusun sebagai **lembar panduan ringkas (*Cheat Sheet / Speaking Points*)** untuk memandu sesi pertemuan Zoom bersama dosen pembimbing atau penguji. Dokumen ini merangkum struktur master repositori, angka metrik hasil riil, alur cerita riset, dan jawaban taktis atas pertanyaan kritis metodologis.

---

## 🏛️ 1. Struktur Berkas Acuan Master (Single Source of Truth)

Tegaskan kepada dosen/penguji bahwa repositori telah ditata secara bersih dan mandiri:
* **Folder Master (`script_thesis/`)**: Memuat **9 berkas notebook master** yang sesuai urutan pengerjaan (*Run All Ready* & *Pre-rendered Outputs*):
  1. `01_data_processing.ipynb` — Pipeline Data-Centric AI (LLM, Regex, Slang Norm) -> Data v2
  2. `02_lstm_imbalance.ipynb` — LSTM Baseline Alami (Imbalance Data, Save/Load Pickle)
  3. `03_lstm_class_weight.ipynb` — LSTM + Cost-Sensitive Class Weight
  4. `04_lstm_oversampling.ipynb` — LSTM + Random Over-Sampling (ROS)
  5. `05_lstm_undersampling.ipynb` — LSTM + Random Under-Sampling (RUS)
  6. `06_lstm_smote.ipynb` — LSTM + SMOTE (Synthetic Sequence Feature)
  7. `07_indobert_lora.ipynb` — IndoBERTweet-LoRA Vanilla Adapter
  8. `08_tapt_indobert_lora.ipynb` — TAPT IndoBERTweet-LoRA (MLM 3 Epoch + Downstream)
  9. `09_summary_model.ipynb` — Rangkuman Master Dinamis & Plot 4-Panel
* **Poin Plus untuk Disampaikan**: Pembimbing/penguji dapat langsung melihat grafik WordCloud, log epoch, tabel evaluasi Train vs Test, serta **Heatmap Confusion Matrix** di setiap file `.ipynb` tanpa harus menjalankan komputasi ulang dari awal.

---

## 🎙️ 2. Alur Cerita Presentasi Zoom (The 4-Step Storyline)

### Langkah 1: Kualitas Data & Pra-pemrosesan (`01_data_processing.ipynb`)
- **Latar Masalah**: Data mentah hasil scraping Twitter ($n = 8.648$) memiliki **402 tweet terpotong** akibat antarmuka Twitter (*UI truncation* seperti teks *"Tampilkan lebih banyak"*, `...`, broken `t.co`).
- **Solusi Data-Centric AI**: Dilakukan rekonstruksi kontekstual menggunakan LLM untuk melengkapi kalimat yang terpotong tanpa merusak entitas nama, lokasi, atau tanggal bencana.
- **Pembersihan Berurutan**:
  - Regex menghapus URL dan mention; tagar diubah formatnya (`#banjir` $\rightarrow$ `banjir`).
  - Normalisasi kata slang menggunakan kamus leksikon resmi 4.334 kata (`kamus/colloquial-indonesian-lexicon.csv`).
  - Emoticon dikonversi ke kata sentimen kontekstual (😭 $\rightarrow$ sedih, 🙏 $\rightarrow$ doa).
  - Visualisasi **WordCloud 4 panel** mengonfirmasi dominasi leksikon kedaruratan pada tiap kelas sentimen.

### Langkah 2: Eksperimen LSTM & 5 Strategi Penyeimbangan Data (`02` s.d. `06`)
- Seluruh model diuji pada **Hold-out Test Set yang Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, Seed 42)**.
- **Temuan Kritis**:
  1. **Baseline (Imbalance)**: Akurasi global 70,92%, namun Recall kelas Netral (minoritas) sangat rendah (**28,81%**) akibat fenomena *The Accuracy Paradox* & *Majority Collapse*.
  2. **Class Weight (Cost-Sensitive)**: Memberikan penalti bobot loss terbalik; berhasil mendongkrak Recall Netral ke **40,40%** dan Macro F1 ke **64,60%** tanpa manipulasi sampel.
  3. **Random Oversampling (ROS)**: Menghasilkan Recall Netral terbaik di keluarga LSTM (**48,68%**) dan Macro F1 **64,89%** karena menjaga kekayaan kosakata korpus.
  4. **Random Undersampling (RUS)**: Akurasi anjlok drastis ke **53,06%** karena pemangkasan >45% data mayoritas menyebabkan hilangnya wawasan bahasa (*information loss*).
  5. **SMOTE**: Akurasi tertahan di **64,39%** karena interpolasi numerik membangkitkan token sintetis acak tanpa makna semantik pada ruang integer diskrit.

### Langkah 3: Superioritas IndoBERTweet-LoRA & TAPT (`07` dan `08`)
- **IndoBERTweet-LoRA Vanilla (`07`)**: Melompat signifikan ke **Akurasi 77,98%** dan **Macro F1 73,90%** (Recall Netral mencapai **61,26%**).
  - *Mengapa?* Mekanisme *Bidirectional Self-Attention* menangkap konteks kalimat media sosial jauh lebih utuh dibanding pemrosesan sekuensial satu arah LSTM.
  - *Efisiensi LoRA*: Hanya melatih **~0,47% parameter** ($r=16, \alpha=32$), menghemat memori tanpa mengorbankan kapasitas representasi.
- **TAPT IndoBERTweet-LoRA (`08`) — Juara Terbaik Mutlak**:
  - Menerapkan *Task-Adaptive Pretraining (TAPT)* berupa Masked Language Modeling (MLM 3 epoch, $lr=5 \times 10^{-5}$) pada korpus tweet banjir sebelum fine-tuning klasifikasi hilir.
  - Hasil: Menembus rekor tertinggi **Akurasi 80,06%** dan **Macro F1 74,61%** (model pertama dan satu-satunya yang menembus batas psikologis 80%).
  - *Mengapa efektif?* TAPT mengadaptasi bobot enkoder terhadap istilah hidrologis kebencanaan lokal Sumatra sebelum proses supervisi klasifikasi dimulai.

---

## 📊 3. Tabel Master Metrik (Hafalan Cepat Saat Ditanya)

| No | Model / Notebook | Strategi Penanganan | Train Acc (%) | Test Acc (%) | Macro F1 (%) | Generalization Gap Acc (%) | Recall Netral (Minoritas) | Keterangan Kinerja |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **02. LSTM Baseline** | Natural (Imbalance) | 81.32% | **70.92%** | **61.56%** | +10.40% | 28.81% | Bias mayoritas (*Majority Collapse*) |
| 2 | **03. LSTM Class Weight** | Cost-Sensitive Loss | 88.48% | **71.68%** | **64.60%** | +16.81% | 40.40% | Recall netral naik tajam (+11.59%) |
| 3 | **04. LSTM Oversampling** | ROS (Data Latih) | 92.46% | **70.06%** | **64.89%** | +22.40% | 48.68% | Recall netral tertinggi di LSTM |
| 4 | **05. LSTM Undersampling** | RUS (Data Latih) | 70.59% | **53.06%** | **52.72%** | +17.53% | 56.62% | Anjlok parah (*information loss*) |
| 5 | **06. LSTM SMOTE** | Synthetic Sequence | 65.03% | **64.39%** | **57.37%** | +0.63% | 43.71% | Token diskrit sintetis rusak |
| 6 | **07. IndoBERT-LoRA** | Adapter Tuning ($r=16$) | 84.15% | **77.98%** | **73.90%** | +6.17% | 61.26% | Unggul mutlak atas seluruh LSTM |
| 7 | **08. TAPT IndoBERT-LoRA**| MLM Domain + LoRA | 86.40% | **80.06%** | **74.61%** | +6.34% | 52.65% | **Model Terbaik Keseluruhan (>80%)** |

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
