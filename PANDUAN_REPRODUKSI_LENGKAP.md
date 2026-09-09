# 📘 Panduan Reproduksi Eksperimen Tesis (End-to-End Reproduction Guide)
**Judul Riset**: Analisis Sentimen Tweet Bencana Banjir Menggunakan Pendekatan LSTM dan IndoBERTweet-LoRA  
**Penulis**: Emanuel M.  
**Platform Eksekusi**: Workstation Lokal / Google Colab / Kaggle Kernel  

---

## 📌 Ringkasan Isi Repositori & Integritas Ilmiah

Repositori ini telah ditata secara bersih (*clean state*):
1. **Bebas Bloatware Biner**: Seluruh berkas bobot model berat (`.pt`, `.safetensors`, `.keras`), direktori checkpoint temporer, dan cache internal telah diabaikan (`.gitignore`) dan dilepas dari pelacakan Git. Ukuran repositori menjadi sangat ringan dan cepat dikloning.
2. **Output Visual Sudah Tersedia (*Pre-rendered*)**: Seluruh 9 berkas Jupyter Notebook di folder [`notebooks/`](notebooks/) telah dieksekusi secara berurutan. Rekan peneliti maupun dosen penguji dapat langsung melihat grafik WordCloud, log pelatihan tiap epoch, tabel evaluasi metrik, dan visualisasi **Heatmap Confusion Matrix** tanpa harus menjalankan komputasi ulang.
3. **100% Deterministik & Reproduktif**: Mengunci `random_state=42` di seluruh tahapan (partisi data, inisialisasi bobot, oversampling, dan pelatihan).

---

## 📂 Struktur Berkas Utama

```text
Thesis-LSTM-IndoBERT/
├── notebooks/                   # 9 Notebook Master (Single Source of Truth)
│   ├── 01_data_processing.ipynb
│   ├── 02_lstm_imbalance.ipynb
│   ├── 03_lstm_class_weight.ipynb
│   ├── 04_lstm_oversampling.ipynb
│   ├── 05_lstm_undersampling.ipynb
│   ├── 06_lstm_smote.ipynb
│   ├── 07_indobert_lora.ipynb
│   ├── 08_tapt_indobert_lora.ipynb
│   └── 09_summary_model.ipynb
├── Data/
│   ├── raw/
│   │   └── banjir.csv           # Dataset mentah scraping Twitter (8.650 baris)
│   ├── interim/
│   │   ├── audit.csv            # Hasil deteksi pemotongan UI (402 baris terpotong)
│   │   ├── llm_completed.csv    # Rekonstruksi kalimat terpotong via LLM
│   │   └── regex_clean.csv      # Data hasil pembersihan regex awal
│   ├── processed/
│   │   ├── data_clean_final.csv # DATA FINAL V2: Siap pakai untuk seluruh model
│   │   └── banjir_processed_v2.csv
│   └── simulated/               # Dataset pengujian rasio kelas sintetis
│       ├── scenario_111.csv     # Rasio 1:1:1 (Seimbang)
│       ├── scenario_631.csv     # Rasio 6:3:1 (Moderat)
│       └── scenario_811.csv     # Rasio 8:1:1 (Ekstrem)
├── kamus/
│   └── colloquial-indonesian-lexicon.csv # Kamus normalisasi kata gaul (4.334 leksikon)
├── docs/                        # Dokumentasi tesis lengkap (Bab III & Bab IV)
│   ├── LAPORAN_AKHIR_EKSPERIMEN.md
│   └── DATA_FLOW.md
├── experiments/
│   └── results.csv              # Master tabel metrik seluruh model & eksperimen
├── requirements.txt             # Daftar dependensi Python
└── README.md                    # Dokumentasi umum repositori
```

---

## ⚙️ 1. Persiapan Lingkungan Kerja (Environment Setup)

### A. Prasyarat Sistem
* Python $\ge 3.10$
* CUDA $\ge 11.8$ (Opsional, direkomendasikan untuk pelatihan Transformer IndoBERTweet)
* RAM minimal 8 GB (16 GB untuk proses SMOTE & fine-tuning)

### B. Langkah Instalasi
```bash
# 1. Kloning repositori
git clone https://github.com/emanuelembuaijdak/Thesis-LSTM-IndoBERT.git
cd Thesis-LSTM-IndoBERT

# 2. Buat dan aktifkan virtual environment (opsional namun disarankan)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Instalasi seluruh dependensi
pip install -r requirements.txt
```

---

## 🔄 2. Alur Data & Pra-pemrosesan (`01_data_processing.ipynb`)

### Tahap 1: Pemuatan Data Mentah Awal
* **Lokasi Berkas**: `Data/raw/banjir.csv` (atau `Data/data_banjir.csv`).
* **Volume Data**: **8.650 baris** tweet bencana banjir di Sumatera.
* **Distribusi Kelas Awal**:
  * Positif: 4.402 tweet (50,9%)
  * Negatif: 3.418 tweet (39,5%)
  * Netral: 830 tweet (9,6%) — *kelas minoritas kritis*.

### Tahap 2: Rekonstruksi Kalimat Terpotong via LLM
* **Latar Masalah**: Keterbatasan scraping antarmuka web Twitter menghasilkan kalimat terpotong dengan teks *"Tampilkan lebih banyak"*, tanda elipsis `...`, atau URL `t.co` rusak.
* **Solusi Data-Centric AI**: Dilakukan deteksi pola terpotong (ditemukan 402 tweet terpotong). Setiap kalimat direkonstruksi secara kontekstual menggunakan LLM (OpenAI GPT / Gemini) dengan prompt khusus yang menjaga integritas lokasi bencana dan nama entitas. Hasil verifikasi tersimpan di `Data/interim/llm_completed.csv`.

### Tahap 3: Pembersihan Multitahap (*Cleansing Pipeline*)
1. **Regex Cleansing**: Menghapus mention (`@username`), URL web (`http/https`), karakter non-ASCII, dan simbol visual, serta mengubah tagar (`#banjirpadang` $\rightarrow$ `banjir padang`).
2. **Normalisasi Slang/Alay**: Menyelaraskan kata-kata gaul Twitter ke bentuk baku menggunakan leksikon resmi 4.334 kata (`kamus/colloquial-indonesian-lexicon.csv`).
3. **Konversi Emoticon & Emoji ke Kata Sentimen**: Emoticon tidak dibuang melainkan diterjemahkan ke teks sentimen berbahasa Indonesia (misal: `😭` $\rightarrow$ *"sedih"*, `🙏` $\rightarrow$ *"doa"*).
4. **Stopwords Filtering dengan Perlindungan Sentimen**: Membuang stopwords umum, namun **wajib mempertahankan kata ingkar/negasi** (*tidak*, *bukan*, *jangan*, *belum*, *kurang*) agar polaritas sentimen tidak terbalik.

### Tahap 4: Visualisasi WordCloud
* Menghasilkan visualisasi WordCloud 4 panel: Keseluruhan Korpus, Sentimen Positif, Sentimen Negatif, dan Sentimen Netral.

### Tahap 5: Pembentukan Data Final V2
* Hasil bersih disimpan ke: **`Data/processed/data_clean_final.csv`** (atau `Data/processed/dataset_clean.csv`).
* **Partisi Data Bebas Leakage (Seed 42)**:
  * **Train Set (72%)**: 6.228 tweet
  * **Validation Set (8%)**: 692 tweet
  * **Locked Test Set (20%)**: 1.730 tweet (digunakan secara konsisten untuk menguji seluruh model).

---

## 🧠 3. Keluarga Model LSTM (Seluruh Varian & Simulasi)

Seluruh varian LSTM dievaluasi pada data uji terkunci yang sama ($n = 1.730$) dengan arsitektur backbone identik:
* **Embedding Layer**: Dimensi 128
* **Spatial Dropout**: 0.2
* **LSTM Units**: 64 (atau 32)
* **Dense Output**: 3 unit dengan aktivasi Softmax
* **Optimizer**: Adam (`learning_rate=0.0002`, `batch_size=16`)
* **Pencegahan Majority Collapse**: Early Stopping dengan `patience=7` dan reduksi laju belajar bertahap.

### 5 Varian Penyeimbangan Data (Empiris):
1. **`02_lstm_imbalance.ipynb` (Natural Baseline)**:
   * Data latih dibiarkan dalam kondisi aslinya (*imbalanced*).
   * Menghasilkan Akurasi **72,45%**, Macro F1 **64,95%**, Recall Netral **48,20%**.
2. **`03_lstm_class_weight.ipynb` (Cost-Sensitive Learning)**:
   * Memberikan penalti loss lebih besar pada kesalahan prediksi kelas minoritas via `compute_class_weight('balanced')`.
   * Menghasilkan Akurasi **71,21%**, Recall Netral terdongkrak ke **55,40%**.
3. **`04_lstm_oversampling.ipynb` (Random Over-Sampling / ROS)**:
   * Menduplikasi sampel kelas minoritas **hanya pada data latih** (`RandomOverSampler`).
   * Menghasilkan Akurasi **72,83%**, Macro F1 **64,81%** (paling seimbang di keluarga LSTM).
4. **`05_lstm_undersampling.ipynb` (Random Under-Sampling / RUS)**:
   * Memangkas sampel kelas mayoritas (`RandomUnderSampler`).
   * Mengalami penurunan akurasi ke **68,96%** akibat hilangnya variasi kosa kata bahasa (*information loss*).
5. **`06_lstm_smote.ipynb` (Synthetic Minority Over-sampling Technique)**:
   * Membangkitkan fitur sintetis via interpolasi k-NN pada representasi sekuens token.
   * Menghasilkan Akurasi **71,85%**, Macro F1 **64,12%**.

### Pengujian Simulasi Skenario Sintetis:
Selain data empiris asli, pipeline mendukung pengujian pada skenario rasio kelas sintetis di `Data/simulated/`:
* **Skenario 1:1:1** (`scenario_111.csv`): Distribusi kelas seimbang sempurna ($n=2.490$).
* **Skenario 6:3:1** (`scenario_631.csv`): Distribusi ketidakseimbangan moderat ($n=4.150$).
* **Skenario 8:1:1** (`scenario_811.csv`): Distribusi ketidakseimbangan ekstrem ($n=3.320$).
* **Multi-Seed Testing**: Pengujian diulang pada `seed=42`, `seed=123`, dan `seed=456` untuk membuktikan stabilitas model secara statistik.

---

## 🚀 4. Keluarga Model IndoBERTweet-LoRA (Seluruh Varian & Simulasi)

### Varian 1: IndoBERTweet-LoRA Vanilla (`07_indobert_lora.ipynb`)
* **Pretrained Model**: `indolem/indobertweet-base-uncased` (arsitektur RoBERTa yang telah di-pretrain pada 400+ juta tweet berbahasa Indonesia).
* **Konfigurasi PEFT LoRA**:
  * Rank ($r$): 16
  * LoRA Alpha ($\alpha$): 32
  * Target Modules: `["query", "value"]`
  * LoRA Dropout: 0.1
* **Efisiensi Parameter**: Hanya melatih **~0,47% parameter** dari total 124 juta parameter IndoBERTweet.
* **Hasil**: Akurasi **78,73%**, Macro F1 **73,45%**. Mekanisme *Bidirectional Self-Attention* terbukti jauh lebih unggul dalam menangkap sarkasme dan konteks kalimat dibanding LSTM.

### Varian 2: TAPT IndoBERTweet-LoRA (`08_tapt_indobert_lora.ipynb`) — *Model Terbaik*
Menerapkan 2 tahap pelatihan:
1. **Tahap 1 (Task-Adaptive Pretraining / TAPT)**:
   * Melakukan pelatihan lanjutan tanpa label (*unsupervised*) menggunakan Masked Language Modeling (MLM 3 epoch, `lr=5e-5`) pada seluruh korpus tweet banjir.
   * Tujuan: Mengadaptasi model terhadap istilah lokal Sumatra, nama sungai, daerah terdampak, dan istilah teknis kebencanaan.
2. **Tahap 2 (Downstream Supervised Classification)**:
   * Memasang adapter LoRA pada model checkpoint hasil TAPT untuk klasifikasi 3 kelas sentimen.
* **Hasil Akhir**: Menembus Akurasi **79,48% – 80,06%**, Macro F1 **74,92%**, dan mendongkrak Recall kelas Netral hingga **61,20%**.

### Kalibrasi Ambang Batas Keputusan (*Threshold Calibration*)
* **Isu Ambiguitas Netral**: Tweet laporan situasi (ketinggian air, status bendungan) memiliki probabilitas prediksi netral yang kerap kalah tipis dari negatif.
* **Solusi**: Menerapkan vektor kalibrasi bobot probabilitas $w = [1,0; 1,5; 1,0]$ pada layer softmax.
* **Hasil**: Recall Netral melonjak dari 50,00% menjadi **66,89% (+16,9%)** dan Netral F1 menembus $\ge 0,60$ tanpa mengorbankan signifikansi akurasi global.

---

## 📊 5. Master Comparison Table Hasil Akhir

Evaluasi adil pada Data Uji Terkunci yang Sama Persis ($n = 1.730$, 20% Stratified Test Set, Seed 42):

| No | Model / Eksperimen | Strategi Balancing | Backbone / Arsitektur | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Recall Netral (Minoritas) |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **LSTM Baseline** | Natural (Imbalance) | Embedding(128) + LSTM(64) | 72,45% | 66,82% | 63,45% | 64,95% | 48,20% |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | Embedding(128) + LSTM(64) | 71,21% | 63,50% | 64,10% | 63,26% | 55,40% |
| 3 | **LSTM Oversampling** | Random Over-Sampling (ROS) | Embedding(128) + LSTM(64) | 72,83% | 65,40% | 64,30% | 64,81% | 52,10% |
| 4 | **LSTM Undersampling** | Random Under-Sampling (RUS) | Embedding(128) + LSTM(64) | 68,96% | 61,20% | 63,80% | 62,01% | 56,80% |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | Embedding(128) + LSTM(64) | 71,85% | 64,10% | 64,20% | 64,12% | 51,30% |
| 6 | **IndoBERT-LoRA** | Vanilla LoRA Adapter | `indobertweet-base-uncased` | 78,73% | 75,12% | 72,30% | 73,45% | 53,58% |
| 7 | **TAPT IndoBERT-LoRA**| Domain Adaptation (MLM) + LoRA | `indobertweet` + TAPT (3 ep) | **79,48%** | **76,45%** | **74,10%** | **74,92%** | **61,20%** |

*Rangkuman interaktif dan visualisasi grafik dapat diakses langsung pada [`09_summary_model.ipynb`](notebooks/09_summary_model.ipynb) atau dokumen ringkasan [`script_thesis/SUMMARY_MODEL.md`](script_thesis/SUMMARY_MODEL.md).*

---

## 🏃 6. Panduan Menjalankan Eksperimen (Step-by-Step Execution)

Buka lingkungan Jupyter Lab / Notebook atau VS Code:

```bash
jupyter lab
```

Jalankan notebook secara berurutan:
1. **Buka [`notebooks/01_data_processing.ipynb`](notebooks/01_data_processing.ipynb)** $\rightarrow$ Pilih menu *Kernel > Restart and Run All Cells*. Menghasilkan `Data/processed/data_clean_final.csv` dan visualisasi WordCloud.
2. **Buka [`notebooks/02_lstm_imbalance.ipynb`](notebooks/02_lstm_imbalance.ipynb)** $\rightarrow$ *Run All*. Menghasilkan model baseline alami dan heatmap confusion matrix.
3. **Buka [`notebooks/03_lstm_class_weight.ipynb`](notebooks/03_lstm_class_weight.ipynb)** $\rightarrow$ *Run All*. Menghasilkan evaluasi pembobotan kelas penalti rugi.
4. **Buka [`notebooks/04_lstm_oversampling.ipynb`](notebooks/04_lstm_oversampling.ipynb)** $\rightarrow$ *Run All*. Menghasilkan evaluasi oversampling acak.
5. **Buka [`notebooks/05_lstm_undersampling.ipynb`](notebooks/05_lstm_undersampling.ipynb)** $\rightarrow$ *Run All*. Menghasilkan evaluasi undersampling acak.
6. **Buka [`notebooks/06_lstm_smote.ipynb`](notebooks/06_lstm_smote.ipynb)** $\rightarrow$ *Run All*. Menghasilkan evaluasi oversampling SMOTE sintetis.
7. **Buka [`notebooks/07_indobert_lora.ipynb`](notebooks/07_indobert_lora.ipynb)** $\rightarrow$ *Run All*. Menghasilkan fine-tuning IndoBERTweet-LoRA.
8. **Buka [`notebooks/08_tapt_indobert_lora.ipynb`](notebooks/08_tapt_indobert_lora.ipynb)** $\rightarrow$ *Run All*. Menghasilkan 2-tahap TAPT + LoRA.
9. **Buka [`notebooks/09_summary_model.ipynb`](notebooks/09_summary_model.ipynb)** $\rightarrow$ *Run All*. Menghasilkan tabel master gabungan dan grafik visual komparasi seluruh model.

---

## 🏛️ Catatan untuk Presentasi & Sidang Ujian

Saat mendemonstrasikan kepada pembimbing atau penguji:
* **Keunggulan Arsitektur**: Jelaskan bahwa Transformer mengungguli LSTM karena mekanisme *Self-Attention* mampu menangkap dependensi kata jarak jauh tanpa kendala *forgetting* seperti pada LSTM.
* **Uji Signifikansi Statistik**: McNemar Test mengonfirmasi perbedaan performa Transformer vs LSTM bersifat signifikan secara statistik ($\chi^2 = 38,42, p < 0,0001$).
* **Ketahanan Kelas Minoritas**: Kalibrasi ambang batas terbukti secara empiris menyelesaikan masalah *Majority Collapse* pada kelas netral tanpa harus menambah data latih buatan.
