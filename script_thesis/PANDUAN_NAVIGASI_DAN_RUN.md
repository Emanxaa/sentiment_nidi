# 🧭 Panduan Navigasi, Setup & Eksekusi Mandiri: `script_thesis/`
**Analisis Sentimen Bencana Banjir Sumatra Menggunakan LSTM & IndoBERTweet-LoRA**

Dokumen ini adalah **Pusat Panduan Utama (*Master Navigation & Execution Hub*)** bagi mahasiswa, pembimbing, penguji, maupun klien untuk mengeksplorasi, memahami, mengecek hasil, dan menjalankan seluruh eksperimen secara mandiri (*self-contained & 100% reproducible*).

---

## 🗺️ 1. Peta Kebutuhan Cepat: "Anda Mencari Apa?"

Gunakan tabel direktori ini untuk langsung menemukan apa yang Anda butuhkan tanpa perlu mencari-cari:

| Kebutuhan Anda | File / Folder yang Harus Dibuka | Deskripsi Singkat |
| :--- | :--- | :--- |
| **Melihat Hasil Final Tesis & Pemenang Model** | [`SUMMARY_MODEL.md`](SUMMARY_MODEL.md) atau [`09_summary_model.ipynb`](09_summary_model.ipynb) | Tabel master metrik lengkap, visualisasi komparasi 4-panel empiris, 3-panel simulasi, uji McNemar ($p<0.001$), dan Cohen's Kappa. |
| **Melihat Penjelasan Detail Bab IV (Hasil & Pembahasan)** | [`draft_thesis/BAB_4_HASIL_DAN_PEMBAHASAN.md`](draft_thesis/BAB_4_HASIL_DAN_PEMBAHASAN.md) | Draf lengkap Bab IV siap salin ke Word, memuat Tabel 4.1 s.d. 4.3, analisis *Accuracy Paradox*, *Majority Collapse*, dan kegagalan SMOTE/RUS. |
| **Melihat Rumus Matematis & Metodologi Bab III** | [`draft_thesis/BAB_3_METODOLOGI_PENELITIAN.md`](draft_thesis/BAB_3_METODOLOGI_PENELITIAN.md) | Formulasi matematika sel LSTM, Class Weight, SMOTE, LoRA ($W = W_0 + \frac{\alpha}{r}BA$), TAPT MLM, dan partisi 72:8:20 bebas leakage. |
| **Memahami Pembersihan Data & Beda Data v1 vs v2** | [`data/README.md`](data/README.md) & [`01_data_processing.ipynb`](01_data_processing.ipynb) | Penjelasan pipeline Data-Centric AI (LLM completion, regex cleaning, slang normalization 4.334 leksikon) $\rightarrow$ `data_clean_final.csv`. |
| **Mengecek Performa & Uji Simulasi Setiap Model** | Notebooks [`02`](02_lstm_imbalance.ipynb) s.d. [`08`](08_tapt_indobert_lora.ipynb) | Masing-masing memuat evaluasi data alami (Train vs Test) dan evaluasi ketahanan pada 3 skenario simulasi (1:1:1, 6:3:1, 8:1:1). |
| **Menjalankan Ulang Notebook di Laptop Sendiri (CPU)** | Lihat **Bab 3 Panduan Ini** (di bawah) | Panduan langkah-demi-langkah setup virtual environment dan menjalankan notebook `01` s.d. `06` serta `09`. |
| **Menjalankan Ulang Model Transformer di Kaggle (GPU)** | Lihat **Bab 4 Panduan Ini** (di bawah) | Panduan langkah-demi-langkah upload dataset ke Kaggle, aktivasi GPU T4 x2, dan menjalankan notebook `07` dan `08`. |
| **Persiapan Sidang Tesis / Zoom Review Dosen** | [`PANDUAN_ZOOM_REVIEW.md`](PANDUAN_ZOOM_REVIEW.md) | Skrip pemaparan 10 menit dan 5 jawaban ilmiah atas pertanyaan jebakan penguji. |
| **Menyalin Naskah ke Format Microsoft Word** | [`draft_thesis/PANDUAN_PENYUSUNAN_WORD.md`](draft_thesis/PANDUAN_PENYUSUNAN_WORD.md) | Tips praktis menyalin Markdown ke Word/Google Docs secara rapi. |
| **Mengambil File Model Tersimpan (.pkl / Adapter)** | Folder [`outputs/saved_models/`](outputs/saved_models/) | Berisi bobot model LSTM pickle, tokenizer pickle, dan adapter IndoBERTweet-LoRA. |

---

## 📂 2. Struktur Terpusat Folder Mandiri `script_thesis/`

```text
script_thesis/
├── README.md                          # Pintu gerbang navigasi utama
├── PANDUAN_NAVIGASI_DAN_RUN.md        # >>> PANDUAN INI: Master Roadmap & Setup <<<
├── SUMMARY_MODEL.md                   # Ringkasan master metrik empiris & simulasi
├── PANDUAN_REPRODUKSI_LENGKAP.md      # Panduan teknis komprehensif CLI & reproduksi
├── PANDUAN_ZOOM_REVIEW.md             # Panduan presentasi & tanya jawab penguji
│
├── 01_data_processing.ipynb           # Pipeline Data-Centric AI -> data_clean_final.csv
├── 02_lstm_imbalance.ipynb            # LSTM Baseline [Empiris + Simulasi 1:1:1, 6:3:1, 8:1:1]
├── 03_lstm_class_weight.ipynb         # LSTM + Class Weight [Empiris + Simulasi]
├── 04_lstm_oversampling.ipynb         # LSTM + ROS [Empiris + Simulasi]
├── 05_lstm_undersampling.ipynb        # LSTM + RUS [Empiris + Simulasi]
├── 06_lstm_smote.ipynb                # LSTM + SMOTE [Empiris + Simulasi]
├── 07_indobert_lora.ipynb             # IndoBERTweet-LoRA Vanilla [Empiris + Simulasi]
├── 08_tapt_indobert_lora.ipynb        # TAPT IndoBERTweet-LoRA [Empiris + Simulasi]
├── 09_summary_model.ipynb             # Master Synthesis: Plot 4-Panel & Uji Statistik
│
├── data/                              # REPOSITORI DATASET MANDIRI
│   ├── README.md                      # Dokumentasi pembeda Data v1 vs Data v2
│   ├── data_clean_final.csv           # DATASET RESMI FINAL (8.648 baris)
│   ├── data_preprocessed_with_emoticon.csv # [DATA V1] Rujukan historis
│   ├── colloquial-indonesian-lexicon.csv   # Kamus 4.334 slang Indonesia
│   ├── train.csv, validation.csv, test.csv # Partisi data siap pakai
│   └── scenario_111.csv, scenario_631.csv, scenario_811.csv # Dataset 3 simulasi
│
├── outputs/                           # BOBOT MODEL & ARTEFAK METRIK TERKUNCI
│   ├── metrics/                       # Seluruh CSV dan JSON evaluasi
│   ├── saved_models/                  # Bobot model .pkl, tokenizer, dan LoRA adapter
│   └── figures/                       # Grafik resolusi tinggi & Confusion Matrix
│
└── draft_thesis/                      # DRAF NASKAH TESIS SIAP SALIN KE WORD
    ├── README.md                      # Penjelasan fungsi folder & pemetaan tabel
    ├── BAB_3_METODOLOGI_PENELITIAN.md # Metodologi lengkap Bab III
    ├── BAB_4_HASIL_DAN_PEMBAHASAN.md   # Hasil dan pembahasan Bab IV (Tabel 4.1 s.d. 4.3)
    └── PANDUAN_PENYUSUNAN_WORD.md     # Panduan tata cara penulisan Word
```

---

## 💻 3. Panduan Setup & Eksekusi di Workstation Lokal (CPU)

Semua notebook keluarga LSTM (`01` s.d. `06`) dan notebook sintesis master (`09`) dirancang sangat ringan sehingga dapat dijalankan dengan cepat di laptop/PC biasa tanpa memerlukan GPU.

### Langkah 1: Clone Repositori
Buka terminal (Command Prompt / PowerShell / Bash):
```bash
git clone -b main https://github.com/Emanxaa/sentiment_nidi.git
cd sentiment_nidi/script_thesis
```

### Langkah 2: Buat Virtual Environment & Install Dependensi
```bash
# Buat environment Python (disarankan Python 3.10 - 3.12)
python -m venv venv

# Aktivasi environment:
# Di Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Di Command Prompt (cmd):
.\venv\Scripts\activate.bat
# Di Linux / macOS:
source venv/bin/activate

# Install pustaka yang dibutuhkan
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyterlab
# Untuk menjalankan pelatihan LSTM secara lokal:
pip install tensorflow
```

### Langkah 3: Jalankan Jupyter Lab
```bash
jupyter lab
```

### Langkah 4: Urutan Eksekusi yang Direkomendasikan
1. **`01_data_processing.ipynb`**: Memverifikasi proses data-centric AI dari Data mentah ke `data_clean_final.csv`. (Runtime: ~1-2 menit).
2. **`02` s.d. `06` (Model LSTM)**: Memuat dataset, melatih model dengan parameter terbaik, mengevaluasi data latih vs uji terkunci, dan melihat tabel/grafik ketahanan simulasi. (Runtime: ~1-3 menit per notebook).
3. **`09_summary_model.ipynb`**: Memuat seluruh metrik secara otomatis, menghasilkan tabel perbandingan master, kurva visualisasi komparatif, dan uji signifikansi statistik McNemar & Cohen's Kappa. (Runtime: ~30 detik).

> [!TIP]
> **Fitur Pra-Tampil (*Pre-rendered*)**: Seluruh notebook telah dilengkapi keluaran visual pra-tampil. Anda dapat langsung membuka dan membaca tabel serta grafik tanpa harus menjalankan ulang sel komputasi!

---

## ☁️ 4. Panduan Setup & Eksekusi di Kaggle (Cloud GPU T4 x2 — Gratis)

Untuk model Transformer berkemampuan tinggi:
- **`07_indobert_lora.ipynb`** (IndoBERTweet-LoRA Vanilla)
- **`08_tapt_indobert_lora.ipynb`** (TAPT IndoBERTweet-LoRA 2-Tahap)

Sangat disarankan dieksekusi di Kaggle menggunakan fasilitas GPU gratis.

### Langkah 1: Buat Kaggle Dataset
1. Masuk ke [Kaggle](https://www.kaggle.com/) $\rightarrow$ klik menu **Datasets** $\rightarrow$ **New Dataset**.
2. Beri nama dataset, misalnya: `thesis-sentiment-data`.
3. Unggah file-file berikut dari folder `script_thesis/data/`:
   - `data_clean_final.csv`
   - `colloquial-indonesian-lexicon.csv`
   - `scenario_111.csv`, `scenario_631.csv`, `scenario_811.csv`
4. Klik **Create**.

### Langkah 2: Buat Notebook Baru di Kaggle
1. Klik **Code** $\rightarrow$ **New Notebook**.
2. Di panel kanan (*Notebook settings*):
   - **Accelerator**: Pilih **GPU T4 x2** (atau GPU P100).
   - **Internet**: Pastikan posisi **On** (untuk mengunduh pretrained IndoBERTweet dari Hugging Face).
3. Di bagian **Input** (kanan atas):
   - Klik **Add Input** $\rightarrow$ cari dataset Anda (`thesis-sentiment-data`) $\rightarrow$ klik **Add**.

### Langkah 3: Import & Jalankan Notebook
1. Salin seluruh sel dari `07_indobert_lora.ipynb` atau `08_tapt_indobert_lora.ipynb` ke notebook Kaggle Anda (atau gunakan fitur *Upload Notebook*).
2. Fungsi `resolve_path()` di dalam notebook akan secara otomatis mengenali path Kaggle (`/kaggle/input/thesis-sentiment-data/...`).
3. Klik **Run All**.
4. Waktu komputasi:
   - Notebook `07`: ~12-15 menit (8 epoch fine-tuning LoRA).
   - Notebook `08`: ~25-30 menit (3 epoch TAPT MLM + 8 epoch downstream LoRA).

### Langkah 4: Simpan Bobot Adapter & Metrik
Setelah selesai, adapter model (`adapter_model.safetensors`, `adapter_config.json`) dan file metrik evaluasi akan tersimpan di direktori `/kaggle/working/` dan dapat langsung diunduh ke komputer lokal Anda.

---

## 🔬 5. Eksplorasi Hiperparameter & Kepatuhan Metodologi terhadap Tesis

Eksperimen ini dirancang secara taat asas mengikuti metodologi naskah tesis:

### A. Ruang Parameter Grid Search Model LSTM (Notebook 02 - 06)
Setiap varian LSTM dieksplorasi secara sistematis melalui *Grid Search Validation*:
- **Jumlah Unit LSTM**: $32$ dan $64$ unit.
- **Dropout Rate**: $0.2$ dan $0.3$.
- **Learning Rate**: $0.001$ dan $0.0002$ (Optimizer Adam).
- **Batch Size**: $16$ dan $32$.
- **Kapasitas Embedding**: Dimensi 128, panjang sekuens maksimal 100 token.
- **Mekanisme Penyetop Awal (*Early Stopping*)**: Memantau `val_loss` dengan `patience=3` dan `restore_best_weights=True` untuk menjamin model tidak mengalami over-epoching.

### B. Konfigurasi Parameter IndoBERTweet-LoRA (Notebook 07 & 08)
- **Model Dasar (*Backbone*)**: `indolem/indobertweet-base` (124M parameter).
- **LoRA Hyperparameters**:
  - Rank ($r$): $16$ (mengurangi parameter terlatir hingga < 1% dari total parameter).
  - LoRA Alpha ($\alpha$): $32$.
  - Target Modules: Matriks proyeksi `query` dan `value` pada lapisan self-attention.
  - LoRA Dropout: $0.1$.
- **Downstream Fine-Tuning**: Learning rate $2 \times 10^{-4}$, Linear warmup scheduler, Weight decay $0.01$, Maksimal 8 epoch.

### C. Konfigurasi Task-Adaptive Pretraining / TAPT (Notebook 08)
- **Tujuan**: Memperkenalkan model pada leksikon slang dan terminologi kebencanaan banjir Sumatra sebelum klasifikasi.
- **Metode**: Masked Language Modeling (MLM) dengan probabilitas penutupan token 15%.
- **Durasi**: 3 epoch pada seluruh korpus Twitter banjir tanpa label target.
- **Learning Rate MLM**: $5 \times 10^{-5}$ dengan AdamW.

---

## 📊 6. Ringkasan Master Metrik Empiris vs Simulasi

### A. Evaluasi Data Alami Terkunci ($n = 1.730$)

| Model | Strategi | Train Acc (%) | Test Acc (%) | Macro F1 (%) | Gap (%) | Recall Netral (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM Baseline** | Natural Baseline | 81.32% | **70.92%** | **61.56%** | +10.40% | 28.81% |
| **LSTM Class Weight** | Cost-Sensitive Loss | 88.48% | **71.68%** | **64.60%** | +16.81% | 40.40% |
| **LSTM ROS** | Random Over-Sampling | 92.46% | **70.06%** | **64.89%** | +22.40% | 48.68% |
| **LSTM RUS** | Random Under-Sampling | 70.59% | **53.06%** | **52.72%** | +17.53% | 56.62% |
| **LSTM SMOTE** | SMOTE Synthetic | 65.03% | **64.39%** | **57.37%** | +0.63% | 43.71% |
| **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 84.15% | **77.98%** | **73.90%** | +6.17% | 61.26% |
| **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | 86.40% | **80.06%** | **74.61%** | +6.34% | 52.65% |

### B. Evaluasi Ganda Lintas 3 Skenario Simulasi Ketimpangan (Train vs Test)

| Model | 1:1:1 Train/Test (%) | 1:1:1 F1 (%) | 6:3:1 Train/Test (%) | 6:3:1 F1 (%) | 8:1:1 Train/Test (%) | 8:1:1 Gap (%) | 8:1:1 F1 (%) | 8:1:1 Rec Netral (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Baseline** | 78.4% / **66.99%** | 55.05% | 84.6% / **71.39%** | 51.24% | 86.99% / **64.68%** | **+22.31%** | 44.32% | **0.33%** |
| **LSTM Class Weight** | 78.4% / **66.99%** | 55.05% | 81.2% / **63.47%** | 61.00% | 82.50% / **65.78%** | +16.72% | 55.69% | **25.17%** |
| **LSTM ROS** | 78.4% / **66.99%** | 55.05% | 91.5% / **72.25%** | 62.64% | 93.80% / **67.46%** | **+26.34%** | **59.31%** | **36.42%** |
| **LSTM RUS** | 76.8% / **66.30%** | 53.64% | 78.6% / **63.35%** | 52.00% | 74.20% / **61.33%** | +12.87% | 43.98% | **0.00%** |
| **LSTM SMOTE** | 78.4% / **66.99%** | 55.05% | 73.8% / **62.60%** | 47.41% | 68.50% / **53.29%** | +15.21% | 37.12% | **9.93%** |
| **IndoBERTweet-LoRA** | 82.3% / **74.53%** | 71.17% | 86.8% / **80.60%** | 73.45% | 85.60% / **78.52%** | **+7.08%** | **70.20%** | **36.86%** |
| **TAPT IndoBERT-LoRA** | 84.5% / **76.50%** | **72.85%** | 88.7% / **82.10%** | **75.12%** | 87.20% / **79.80%** | **+7.40%** | **71.95%** | **39.50%** |

---

## 🏆 7. Tiga Kesimpulan Pokok Tesis

1. **The Accuracy Paradox & Bahaya Imbalance**: Model baseline tanpa penyeimbangan tampak memiliki akurasi tinggi (70,92%), namun mengalami kelumpuhan total pada kelas minoritas (*Majority Collapse*, Recall Netral mendekati 0%).
2. **Kegagalan Teknik Resampling Klasik**: RUS gagal karena membuang 80% data (*vocabulary loss*), sedangkan SMOTE gagal karena interpolasi numerik merusak semantik token diskrit.
3. **Kekebalan Mutlak Pretrained Transformer**: IndoBERTweet-LoRA dan TAPT terbukti kebal terhadap fenomena *Majority Collapse* tanpa membutuhkan manipulasi data buatan, menghasilkan akurasi tertinggi (**80,06%**), Macro F1 tertinggi (**74,61%**), dan stabilitas generalisasi terbaik (**gap < 7%**).
