# 🎓 Pusat Kerja Mandiri Tesis: Folder `script_thesis/`

Selamat datang di direktori **`script_thesis/`**. Folder ini dirancang sebagai **satu wadah terpusat (*all-in-one standalone hub*)** yang menampung seluruh artefak riset tesis:
1. **9 Notebook Master** (`01` s.d. `09`) dengan keluaran visual (*pre-rendered outputs*).
2. **Dataset Lengkap** (Data v1, Data v2 bersih final, kamus leksikon slang, data simulasi).
3. **Model Tersimpan** (Pickle Keras model & tokenizer, serta adapter LoRA).
4. **Metrik & Ringkasan Master** (JSON & CSV tabel perbandingan).
5. **Draf Naskah Tesis** (Bab III Metodologi & Bab IV Hasil dan Pembahasan).

---

> [!IMPORTANT]
> **PANDUAN LENGKAP NAVIGASI & EKSEKUSI**:
> Jika Anda atau klien baru pertama kali membuka folder ini, silakan langsung membaca **[`PANDUAN_NAVIGASI_DAN_RUN.md`](PANDUAN_NAVIGASI_DAN_RUN.md)** untuk panduan cepat *"Anda Mencari Apa?"*, petunjuk instalasi lokal (CPU), petunjuk run di Kaggle (GPU), dan pemetaan seluruh file secara lengkap.

---

## 🎯 1. Ringkasan Master Hasil Final (Data Uji Terkunci $n=1.730$)

Seluruh model dievaluasi secara adil pada **Hold-out Test Set yang Sama Persis (20% Stratified Split, Seed 42)**:

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Gap Acc (%) | Train F1 (%) | Test F1 (%) | Gap F1 (%) | Recall Netral (%) | Status & Catatan |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | +10.40% | 73.19% | **61.56%** | +11.63% | 28.81% | *Majority Collapse* pada kelas Netral |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | +16.81% | 85.06% | **64.60%** | +20.46% | 40.40% | Penalti loss; Recall Netral naik (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | +22.40% | 92.44% | **64.89%** | +27.55% | 48.68% | Recall Netral tertinggi di LSTM (Overfitting F1 Gap +27%) |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | +17.53% | 70.64% | **52.72%** | +17.92% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | +0.63% | 64.33% | **57.37%** | +6.96% | 43.71% | Token sintetis merusak semantik diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | +6.17% | 80.20% | **73.90%** | +6.30% | 61.26% | Unggul telak & gap generalisasi F1 sangat sehat (+6.3%) |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | +6.34% | 82.50% | **74.61%** | +7.89% | 52.65% | **JUARA TERBAIK MUTLAK RISET (>80% Acc, 74.6% F1)** |

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

*Visualisasi komparatif 4-panel empiris dan 3-panel simulasi dapat dilihat di notebook [`09_summary_model.ipynb`](09_summary_model.ipynb).*

---


## 🔬 2. Penjelasan Ilmiah Singkat (Bahan Pertanyaan Ujian / Naskah Bab IV)

1. **Mengapa Akurasi LSTM Baseline Lebih Tinggi dari RUS dan SMOTE?**
   * **The Accuracy Paradox**: Kelas mayoritas (Positif & Negatif) mendominasi 82% data uji. Model baseline mudah mencapai skor akurasi tinggi hanya dengan menebak mayoritas, tetapi menderita *Majority Collapse* (Recall Netral hanya **28,81%**).
   * **Information Loss pada RUS**: Pembuangan >45% data mayoritas melenyapkan wawasan kosakata bencana, membuat akurasi anjlok drastis ke **53,06%**.
   * **Discrete Token Corruption pada SMOTE**: Interpolasi numerik pada sekuens integer token menghasilkan representasi non-semantik (kata buatan tak bermakna), menurunkan akurasi ke **64,39%**.
   * **Penyelamatan Minoritas Netral**: Teknik balancing berhasil mendongkrak sensitivitas minoritas: Recall Netral naik tajam ke **40,40% (Class Weight)** dan **48,68% (ROS)**.
2. **Superioritas IndoBERTweet-LoRA vs LSTM**:
   * IndoBERTweet-LoRA melampaui LSTM terbaik dengan selisih akurasi **+9,14%** (70,92% $\rightarrow$ 80,06%) dan Macro F1 **+13,05%** (61,56% $\rightarrow$ 74,61%) berkat mekanisme *Bidirectional Self-Attention*.
3. **Efektivitas Task-Adaptive Pretraining (TAPT)**:
   * Domain adaptation 3 epoch MLM pada korpus banjir berhasil mengadaptasi leksikon kebencanaan lokal Sumatra, mendorong akurasi melampaui ambang batas 80% (**80,06%**).

---

## 📂 3. Struktur Direktori `script_thesis/`

```text
script_thesis/
├── README.md                    # >>> Panduan utama navigasi folder script_thesis <<<
├── SUMMARY_MODEL.md             # Dokumen ringkasan master metrik & analisis Bab IV
├── PANDUAN_REPRODUKSI_LENGKAP.md# Panduan teknis reproduksi end-to-end
├── PANDUAN_ZOOM_REVIEW.md       # Panduan persiapan presentasi & review
│
├── 01_data_processing.ipynb     # Pipeline Data-Centric AI (LLM, Regex, Slang Norm) -> Data v2
├── 02_lstm_imbalance.ipynb      # LSTM Natural Baseline [Empiris + Simulasi 1:1:1, 6:3:1, 8:1:1]
├── 03_lstm_class_weight.ipynb   # LSTM + Cost-Sensitive Class Weight [Empiris + Simulasi]
├── 04_lstm_oversampling.ipynb   # LSTM + Random Over-Sampling (ROS) [Empiris + Simulasi]
├── 05_lstm_undersampling.ipynb  # LSTM + Random Under-Sampling (RUS) [Empiris + Simulasi]
├── 06_lstm_smote.ipynb          # LSTM + SMOTE [Empiris + Simulasi]
├── 07_indobert_lora.ipynb       # IndoBERTweet-LoRA Vanilla Adapter [Empiris + Simulasi]
├── 08_tapt_indobert_lora.ipynb  # TAPT IndoBERTweet-LoRA (MLM 3 Epoch) [Empiris + Simulasi]
├── 09_summary_model.ipynb       # Master Synthesis: Komparasi Empiris & 3 Skenario Simulasi
│
├── data/                        # REPOSITORI DATA
│   ├── README.md                # Panduan perbedaan Data v1 vs Data v2
│   ├── data_clean_final.csv     # >>> DATA V2: DATASET BERSIH FINAL RESMI (8.648 baris) <<<
│   ├── data_preprocessed_with_emoticon.csv # [DATA V1] Dataset lama rujukan historis
│   ├── colloquial-indonesian-lexicon.csv   # Kamus 4.334 leksikon slang bahasa Indonesia
│   ├── train.csv, validation.csv, test.csv # Partisi data siap pakai
│   └── scenario_111.csv, scenario_631.csv, scenario_811.csv # Skenario simulasi
│
├── outputs/                     # HASIL EKSEKUSI & ARTEFAK
│   ├── metrics/                 # File JSON & CSV ringkasan komparasi master
│   │   ├── experiment_metrics_summary.json
│   │   ├── tabel_komparasi_seluruh_model.csv
│   │   └── master_summary.csv
│   ├── saved_models/            # File bobot model tersimpan (.pkl & LoRA adapter)
│   └── figures/                 # Gambar visualisasi WordCloud & Confusion Matrix
│
└── draft_thesis/                # DRAF BAB NASKAH TESIS
    ├── BAB_3_METODOLOGI_PENELITIAN.md
    └── BAB_4_HASIL_DAN_PEMBAHASAN.md
```

---

## 🚀 4. Panduan Menjalankan Notebook

### A. Eksekusi Lokal (CPU Laptop)
Untuk notebook **01** s.d. **06** dan **09**:
```bash
# Dari root repositori
pip install -r requirements.txt
jupyter lab
```
Buka folder `script_thesis/` di Jupyter dan jalankan notebook pilihan Anda. Waktu training per notebook LSTM hanya **1–3 menit**.

### B. Eksekusi Kaggle Cloud GPU (Tesla T4)
Untuk notebook **07** dan **08** (Transformer IndoBERTweet):
1. Buka Kaggle $\rightarrow$ **New Dataset** $\rightarrow$ upload `script_thesis/data/data_clean_final.csv` dan `script_thesis/data/colloquial-indonesian-lexicon.csv`.
2. Buka Kaggle $\rightarrow$ **New Notebook** $\rightarrow$ **File > Import Notebook** $\rightarrow$ pilih `07_indobert_lora.ipynb` atau `08_tapt_indobert_lora.ipynb`.
3. Di panel kanan Kaggle:
   * **Accelerator**: Pilih **GPU T4 x2**.
   * **Internet**: Geser ke **ON**.
   * **Input**: Tambahkan dataset dari Langkah 1.
4. Klik **Run All** atau **Save Version (Commit)** (selesai dalam ~6–8 menit).
5. Unduh output dari tab **Output** Kaggle.

---

## 💾 5. Cara Mengakses Model Tersimpan di Python

```python
import pickle
import pandas as pd

# 1. Memuat Model LSTM
with open("script_thesis/outputs/saved_models/model_lstm_02_baseline_.pkl", "rb") as f:
    model_lstm = pickle.load(f)
with open("script_thesis/outputs/saved_models/tokenizer_lstm_02_baseline_.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 2. Memuat Tabel Komparasi CSV
df_summary = pd.read_csv("script_thesis/outputs/metrics/tabel_komparasi_seluruh_model.csv")
print(df_summary[["Model", "Train Accuracy (%)", "Test Accuracy (%)", "Macro F1 (%)", "Recall Netral (%)"]])
```
