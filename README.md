# 🌊 Thesis-LSTM-IndoBERT: Analisis Sentimen Tweet Bencana Banjir

Repositori penelitian tesis untuk **klasifikasi sentimen 3-kelas (*negatif*, *netral*, *positif*)** pada tweet bencana banjir di Pulau Sumatra dengan membandingkan arsitektur **LSTM (dengan 5 strategi penanganan ketidakseimbangan data)** melawan model **IndoBERTweet-LoRA (dengan Task-Adaptive Pretraining)**.

* **Repository Clone URL**: `https://github.com/Emanxaa/sentiment_nidi.git`
* **Branch Utama**: `main` *(tersedia juga branch backup: `hasil_sabtu`)*

> 📖 **PANDUAN LENGKAP REPRODUKSI**: Untuk panduan langkah demi langkah menjalankan secara lokal dan di Kaggle Cloud GPU, silakan baca **[PANDUAN_REPRODUKSI_LENGKAP.md](PANDUAN_REPRODUKSI_LENGKAP.md)**.

---

## 🎯 1. Ringkasan Master Hasil Final

Seluruh model dievaluasi secara adil pada **Data Uji Terkunci yang Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, Seed 42)**:

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Macro F1 (%) | Generalization Gap Acc (%) | Recall Netral (%) | Status / Catatan |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | **61.56%** | +10.40% | 28.81% | *Majority Collapse* pada kelas Netral |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | **64.60%** | +16.81% | 40.40% | Recall Netral naik signifikan (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | **64.89%** | +22.40% | 48.68% | Recall Netral terbaik di LSTM (+19.87%) |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | **52.72%** | +17.53% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | **57.37%** | +0.63% | 43.71% | Token sintetis merusak semantik diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | **73.90%** | +6.17% | 61.26% | Unggul telak atas seluruh LSTM |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | **74.61%** | +6.34% | 52.65% | **Juara Terbaik Mutlak (>80% Akurasi)** |

### 🧪 Cross-Check Ketahanan Model Lintas 3 Skenario Simulasi: Evaluasi Ganda Training vs Testing

| No | Model | Strategi Balancing | 1:1:1 Train / Test Acc (%) | 1:1:1 F1 (%) | 6:3:1 Train / Test Acc (%) | 6:3:1 F1 (%) | 8:1:1 Train / Test Acc (%) | 8:1:1 Gap Acc (%) | 8:1:1 F1 (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Model |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 78.4% / **66.99%** | 55.05% | 84.6% / **71.39%** | 51.24% | 86.99% / **64.68%** | **+22.31%** | 44.32% | **0.33%** | **Total Majority Collapse** (Overfitting mayoritas) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 78.4% / **66.99%** | 55.05% | 81.2% / **63.47%** | 61.00% | 82.50% / **65.78%** | +16.72% | 55.69% | **25.17%** | **Sangat Tangguh**; Penalti loss mencegah keruntuhan |
| 3 | **LSTM ROS** | Random Over-Sampling | 78.4% / **66.99%** | 55.05% | 91.5% / **72.25%** | 62.64% | 93.80% / **67.46%** | +26.34% | **59.31%** | **36.42%** | **Penyelamat Terbaik LSTM** pada rasio 8:1:1 |
| 4 | **LSTM RUS** | Random Under-Sampling | 76.8% / **66.30%** | 53.64% | 78.6% / **63.35%** | 52.00% | 74.20% / **61.33%** | +12.87% | 43.98% | **0.00%** | **Total Collapse** akibat pemotongan data latih |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 78.4% / **66.99%** | 55.05% | 73.8% / **62.60%** | 47.41% | 68.50% / **53.29%** | +15.21% | 37.12% | **9.93%** | **Gagal**; Vektor interpolasi merusak token diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 82.3% / **74.53%** | 71.17% | 86.8% / **80.60%** | 73.45% | 85.60% / **78.52%** | **+7.08%** | **70.20%** | **36.86%** | **Kebal Collapse** (Gap stabil sehat <8%) |
| 7 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | 84.5% / **76.50%** | **72.85%** | 88.7% / **82.10%** | **75.12%** | 87.20% / **79.80%** | **+7.40%** | **71.95%** | **39.50%** | **JUARA KETAHANAN MUTLAK** (Terkuat di semua rasio) |

*Visualisasi komparatif 4-panel empiris dan 3-panel simulasi dapat dilihat di notebook [`notebooks/09_summary_model.ipynb`](notebooks/09_summary_model.ipynb).*

---


## 🔬 2. Penjelasan Singkat Temuan Ilmiah (Bab IV Tesis)

1. **Mengapa Akurasi LSTM Baseline Lebih Tinggi dari RUS dan SMOTE?**
   * **The Accuracy Paradox**: Kelas mayoritas (Positif & Negatif) mencakup 82% korpus uji. Model baseline mudah mencatat akurasi 70,92% hanya dengan menebak mayoritas, tetapi gagal mendeteksi kelas Netral (Recall Netral hanya **28,81%** / *Majority Collapse*).
   * **Information Loss pada RUS**: Pemangkasan >45% data mayoritas membuang ribuan variasi kata penting sehingga akurasi anjlok ke **53,06%**.
   * **Discrete Token Corruption pada SMOTE**: Interpolasi numerik pada sekuens integer token menghasilkan kata artifisial acak tanpa makna leksikal, menurunkan akurasi ke **64,39%**.
   * **Tujuan Penyeimbangan**: Penyeimbangan berhasil menaikkan sensitivitas minoritas Netral hingga **40,40% (Class Weight)** dan **48,68% (ROS)**.
2. **Superioritas IndoBERTweet-LoRA vs LSTM**:
   * IndoBERTweet-LoRA melampaui LSTM terbaik dengan selisih akurasi **+9,14%** (70,92% $\rightarrow$ 80,06%) dan Macro F1 **+13,05%** (61,56% $\rightarrow$ 74,61%).
   * Mekanisme *Bidirectional Self-Attention* mampu menangkap bahasa gaul, sarkasme, dan relasi kata jarak jauh tanpa *information loss*.
3. **Efektivitas Task-Adaptive Pretraining (TAPT)**:
   * 3 epoch Masked Language Modeling (MLM) pada 8.648 tweet banjir lokal berhasil mengadaptasi leksikon kebencanaan lokal Sumatra, mendorong akurasi melampaui ambang batas 80% (**80,06%**).

---

## 📁 3. Panduan Dataset: Data v1 vs Data v2

| Aspek | Data v1 (Data Lama) | Data v2 (Data Bersih Final Resmi) |
| :--- | :--- | :--- |
| **Lokasi File** | [`Data/data_preprocessed_with_emoticon.csv`](Data/data_preprocessed_with_emoticon.csv) | [`Data/processed/data_clean_final.csv`](Data/processed/data_clean_final.csv) |
| **Jumlah Baris** | 8.650 baris | **8.648 baris** |
| **Kualitas Teks** | Teks terpotong akibat UI Twitter | 402 kalimat terpotong telah direkonstruksi via LLM (`Data/interim/llm_completed.csv`) |
| **Normalisasi** | Pembersihan dasar | Pembersihan regex lengkap + normalisasi leksikon 4.334 kata slang (`kamus/colloquial-indonesian-lexicon.csv`) |
| **Status Pemakaian** | Rujukan eksperimen historis | **Digunakan oleh seluruh notebook model (`02` s.d. `08`)** |

*Dokumentasi detail dataset dapat dibaca pada [`Data/README.md`](Data/README.md).*

---

## 📓 4. Daftar 9 Notebook Master (`notebooks/`)

Setiap notebook mandiri, terurut, dan dilengkapi keluaran visual (*pre-rendered*):

| No | File Notebook | Deskripsi & Tujuan | Device Rekomendasi |
| :---: | :--- | :--- | :---: |
| **01** | [`01_data_processing.ipynb`](notebooks/01_data_processing.ipynb) | Pipeline Data-Centric AI (LLM Completion, Regex, Slang Norm) $\rightarrow$ `data_clean_final.csv` | Lokal (CPU) |
| **02** | [`02_lstm_imbalance.ipynb`](notebooks/02_lstm_imbalance.ipynb) | Model LSTM Baseline Alami (Train vs Test, Save/Load Pickle) | Lokal (CPU) |
| **03** | [`03_lstm_class_weight.ipynb`](notebooks/03_lstm_class_weight.ipynb) | Model LSTM dengan Pembobotan Rugi (*Cost-Sensitive Learning*) | Lokal (CPU) |
| **04** | [`04_lstm_oversampling.ipynb`](notebooks/04_lstm_oversampling.ipynb) | Model LSTM dengan Duplikasi Minoritas (*Random Over-Sampling*) | Lokal (CPU) |
| **05** | [`05_lstm_undersampling.ipynb`](notebooks/05_lstm_undersampling.ipynb) | Model LSTM dengan Pemotongan Mayoritas (*Random Under-Sampling*) | Lokal (CPU) |
| **06** | [`06_lstm_smote.ipynb`](notebooks/06_lstm_smote.ipynb) | Model LSTM dengan Sintesis Token Sekuens (*SMOTE*) | Lokal (CPU) |
| **07** | [`07_indobert_lora.ipynb`](notebooks/07_indobert_lora.ipynb) | Model IndoBERTweet-LoRA Vanilla Adapter ($r=16, \alpha=32$) | Kaggle (Cloud GPU) |
| **08** | [`08_tapt_indobert_lora.ipynb`](notebooks/08_tapt_indobert_lora.ipynb) | Model TAPT IndoBERTweet-LoRA 2-Tahap (MLM 3 Epoch + Downstream) | Kaggle (Cloud GPU) |
| **09** | [`09_summary_model.ipynb`](notebooks/09_summary_model.ipynb) | **Master Summary**: Pemuatan Dinamis JSON, Tabel Master & Visualisasi 4-Panel | Lokal (CPU) |

---

## 🚀 5. Cara Menjalankan Eksperimen

### A. Di Workstation Lokal (CPU)
```bash
# 1. Clone repository
git clone -b main https://github.com/Emanxaa/sentiment_nidi.git
cd sentiment_nidi

# 2. Setup Virtual Environment & Install Dependensi
python -m venv venv
.\venv\Scripts\activate   # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

# 3. Buka Jupyter
jupyter lab
```
Buka folder `notebooks/` dan jalankan notebook `01` s/d `06` serta `09`. Waktu eksekusi rata-rata hanya **1–3 menit per notebook**.

### B. Di Kaggle (Cloud GPU - Gratis)
Untuk notebook **07** dan **08** (Transformer IndoBERTweet):
1. Buat **New Dataset** di Kaggle, upload `Data/processed/data_clean_final.csv` dan `kamus/colloquial-indonesian-lexicon.csv`.
2. Buat **New Notebook** di Kaggle $\rightarrow$ **File > Import Notebook** $\rightarrow$ upload `07_indobert_lora.ipynb` atau `08_tapt_indobert_lora.ipynb`.
3. Di panel kanan (Settings):
   * **Accelerator**: Pilih **GPU T4 x2** atau **GPU P100**.
   * **Internet**: Geser ke **ON**.
   * **Input**: Tambahkan dataset yang diunggah di Langkah 1.
4. Klik **Save Version > Save & Run All (Commit)** (selesai dalam ~6–8 menit).
5. Unduh hasil output dari folder `/kaggle/working/Output/`.

---

## 💾 6. Lokasi Model Tersimpan & Summary Ekspor

* **Model Pickle LSTM**: [`Output/saved_models/`](Output/saved_models/) (`model_lstm_02_baseline_.pkl`, `model_lstm_03_class_weight_.pkl`, dll.).
* **Adapter IndoBERTweet-LoRA**: [`Output/saved_models/tapt_indobert_lora_model/`](Output/saved_models/tapt_indobert_lora_model/).
* **File Tabel Komparasi CSV**:
  * [`Output/predictions/tabel_komparasi_seluruh_model.csv`](Output/predictions/tabel_komparasi_seluruh_model.csv)
  * [`Output/summary/master_summary.csv`](Output/summary/master_summary.csv)
* **File JSON Metrik Dinamis**: [`Output/metrics/experiment_metrics_summary.json`](Output/metrics/experiment_metrics_summary.json).

---

## 📚 Indeks Dokumentasi Riset

| Dokumen | Deskripsi |
| :--- | :--- |
| **[PANDUAN_REPRODUKSI_LENGKAP.md](PANDUAN_REPRODUKSI_LENGKAP.md)** | **Panduan komprehensif replikasi hasil eksperimen secara mendalam** |
| **[Data/README.md](Data/README.md)** | Dokumentasi struktur data dan perbandingan Data v1 vs Data v2 |
| **[SUMMARY_MODEL.md](SUMMARY_MODEL.md)** | Ringkasan metrik master dan pembahasan ilmiah Bab IV Tesis |
| **[docs/MODELS.md](docs/MODELS.md)** | Spesifikasi arsitektur jaringan saraf & hyperparameter tuning |
| **[docs/RESEARCH_METHODOLOGY.md](docs/RESEARCH_METHODOLOGY.md)** | Metodologi penelitian tesis dan panduan replikabilitas |
