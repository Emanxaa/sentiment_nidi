# Thesis-LSTM-IndoBERT: Analisis Sentimen Tweet Bencana Banjir

Implementasi penelitian tesis untuk **klasifikasi sentimen 3-kelas (*negatif*, *netral*, *positif*)** pada tweet bencana banjir di Sumatera menggunakan pendekatan komparatif **LSTM (dengan 5 strategi penanganan ketidakseimbangan data)** dan **IndoBERTweet-LoRA (dengan Task-Adaptive Pretraining)**.

> 📖 **PANDUAN LENGKAP REPRODUKSI**: Untuk mereplikasi hasil eksperimen 100% identik dari data mentah hingga evaluasi akhir, silakan baca **[PANDUAN_REPRODUKSI_LENGKAP.md](PANDUAN_REPRODUKSI_LENGKAP.md)**.

---

## 🎯 Ringkasan Hasil & Model Pemenang

Seluruh model dievaluasi secara adil pada **Data Uji Terkunci yang Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, Seed 42)**:

| No | Nama Model | Strategi Penyeimbangan Data | Arsitektur / Backbone | Test Accuracy | Macro F1 | Recall Netral (Minoritas) | Catatan Kinerja |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural (Imbalance) | Embedding(128) + LSTM(64) | 72.45% | 64.95% | 48.20% | *Majority Collapse* pada Netral |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | Embedding(128) + LSTM(64) | 71.21% | 63.26% | 55.40% | Recall Netral naik signifikan |
| 3 | **LSTM Oversampling** | Random Over-Sampling (ROS) | Embedding(128) + LSTM(64) | 72.83% | 64.81% | 52.10% | Paling seimbang di keluarga LSTM |
| 4 | **LSTM Undersampling** | Random Under-Sampling (RUS) | Embedding(128) + LSTM(64) | 68.96% | 62.01% | 56.80% | Akurasi drop (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE Sequence Token | Embedding(128) + LSTM(64) | 71.85% | 64.12% | 51.30% | Terbatas pada ruang diskrit |
| 6 | **IndoBERT-LoRA** | Vanilla LoRA Adapter | `indobertweet-base-uncased` | 78.73% | 73.45% | 53.58% | Unggul signifikan atas LSTM |
| 7 | **TAPT IndoBERT-LoRA**| Domain Adaptation (MLM) + LoRA | `indobertweet` + TAPT (3 ep) | **79.48%** | **74.92%** | **61.20%** | **Juara Terbaik Mutlak** |

---

## 📓 9 Notebook Master (Single Source of Truth)

Repositori ini menyediakan 9 berkas notebook mandiri di folder [`notebooks/`](notebooks/) yang siap dibuka (*pre-rendered outputs*):

1. **[`01_data_processing.ipynb`](notebooks/01_data_processing.ipynb)**: Rekonstruksi kalimat terpotong via LLM + Regex + Normalisasi leksikon slang (4.334 kata) + Emoticon sentimen + WordCloud 4 panel $\rightarrow$ Menghasilkan Data Final V2 (`data_clean_final.csv`).
2. **[`02_lstm_imbalance.ipynb`](notebooks/02_lstm_imbalance.ipynb)**: Model LSTM Baseline Alami (Imbalance).
3. **[`03_lstm_class_weight.ipynb`](notebooks/03_lstm_class_weight.ipynb)**: Model LSTM dengan Pembobotan Penalti Rugi (*Cost-Sensitive Class Weight*).
4. **[`04_lstm_oversampling.ipynb`](notebooks/04_lstm_oversampling.ipynb)**: Model LSTM dengan Duplikasi Sampel Minoritas (*Random Over-Sampling / ROS*).
5. **[`05_lstm_undersampling.ipynb`](notebooks/05_lstm_undersampling.ipynb)**: Model LSTM dengan Pemangkasan Sampel Mayoritas (*Random Under-Sampling / RUS*).
6. **[`06_lstm_smote.ipynb`](notebooks/06_lstm_smote.ipynb)**: Model LSTM dengan Sintesis Fitur Sekuens Token (*SMOTE*).
7. **[`07_indobert_lora.ipynb`](notebooks/07_indobert_lora.ipynb)**: Model IndoBERTweet-LoRA Vanilla (hanya melatih ~0.47% parameter).
8. **[`08_tapt_indobert_lora.ipynb`](notebooks/08_tapt_indobert_lora.ipynb)**: Model IndoBERTweet-LoRA 2-Tahap dengan *Task-Adaptive Pretraining* (MLM 3 epoch pada tweet bencana banjir).
9. **[`09_summary_model.ipynb`](notebooks/09_summary_model.ipynb)**: Visualisasi grafik bar komparasi seluruh model dan sintesis temuan ilmiah Bab IV Tesis.

---

## 🚀 Panduan Ringkas Menjalankan Repositori (Quick Start)

### 1. Kloning & Instalasi Dependensi
```bash
git clone https://github.com/emanuelembuaijdak/Thesis-LSTM-IndoBERT.git
cd Thesis-LSTM-IndoBERT
pip install -r requirements.txt
```

### 2. Menjalankan Notebook
```bash
jupyter lab
```
Buka folder [`notebooks/`](notebooks/) dan jalankan notebook berurutan dari `01` hingga `09`. Seluruh path data telah terotomatisasi mendeteksi lingkungan kerja lokal, Google Colab, maupun Kaggle.

---

## 📚 Indeks Dokumentasi Riset

| Dokumen | Deskripsi |
|---|---|
| **[PANDUAN_REPRODUKSI_LENGKAP.md](PANDUAN_REPRODUKSI_LENGKAP.md)** | **Panduan teknis utama replikasi hasil eksperimen secara mendalam** |
| [LAPORAN_AKHIR_EKSPERIMEN.md](docs/LAPORAN_AKHIR_EKSPERIMEN.md) | Laporan naskah Bab IV Tesis & analisis komparasi komprehensif |
| [SUMMARY_MODEL.md](script_thesis/SUMMARY_MODEL.md) | Ringkasan metrik master dan temuan arsitektur |
| [PANDUAN_ZOOM_REVIEW.md](script_thesis/PANDUAN_ZOOM_REVIEW.md) | Panduan paparan & tanya-jawab sesi review dengan pembimbing/penguji |
| [DATA_FLOW.md](docs/DATA_FLOW.md) | Alur rekayasa data dari raw hingga data final v2 |
| [AGENTS.md](AGENTS.md) | Standar operasional riset & integritas reproduksibilitas ilmiah |

---

## ⚖️ Integritas Repositori
Seluruh berkas model biner besar (`.pt`, `.safetensors`, `.keras`) dan cache temporer telah dikecualikan melalui [`.gitignore`](.gitignore) agar repositori tetap bersih, ringan, dan sesuai standar publikasi GitHub.

