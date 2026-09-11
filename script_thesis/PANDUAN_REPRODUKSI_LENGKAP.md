# 📘 Panduan Reproduksi Eksperimen Tesis (End-to-End Reproduction Guide)
**Judul Riset**: Analisis Sentimen Tweet Bencana Banjir Menggunakan Pendekatan LSTM dan IndoBERTweet-LoRA  
**Penulis**: Emanuel M.  
**Repository GitHub**: [https://github.com/Emanxaa/sentiment_nidi](https://github.com/Emanxaa/sentiment_nidi) (Branch: `main`)  
**Platform Eksekusi**: Workstation Lokal (CPU) / Kaggle Cloud GPU (Nvidia Tesla T4)  

---

## 📌 1. Ringkasan Repositori & Integritas Ilmiah

Repositori ini ditata agar **siap dikloning dan langsung digunakan oleh siapa pun (dosen, penguji, rekan peneliti, atau klien)** dengan standar:
1. **Bebas Bloatware Biner**: Seluruh file bobot biner raksasa (>100MB) dan cache temporer di-exclude via `.gitignore`. Model yang disimpan berformat ringkas (Pickle Keras ~16MB dan Adapter LoRA ~2MB) sehingga kloning berlangsung cepat dan hemat kuota.
2. **Output Visual Lengkap (*Pre-rendered*)**: Seluruh 9 berkas Jupyter Notebook di folder [`notebooks/`](notebooks/) telah dieksekusi secara nyata (*live raw run*). Anda dapat langsung melihat grafik WordCloud, kurva epoch, tabel evaluasi Train vs Test, dan visualisasi Confusion Matrix tanpa wajib melatih ulang dari nol.
3. **100% Deterministik & Reproduktif**: Kunci seed `SEED = 42` diterapkan seragam pada seluruh tahapan (partisi data, inisialisasi bobot, sampling, dan pelatihan).

---

## 📂 2. Struktur Berkas & Navigasi Cepat

```text
sentiment_nidi/
├── notebooks/                   # 9 Notebook Master (Single Source of Truth)
│   ├── 01_data_processing.ipynb # Pipeline Data-Centric AI (LLM, Regex, Slang Norm) -> Data v2
│   ├── 02_lstm_imbalance.ipynb  # LSTM Natural Baseline (Imbalance Data)
│   ├── 03_lstm_class_weight.ipynb # LSTM + Cost-Sensitive Class Weight
│   ├── 04_lstm_oversampling.ipynb# LSTM + Random Over-Sampling (ROS)
│   ├── 05_lstm_undersampling.ipynb# LSTM + Random Under-Sampling (RUS)
│   ├── 06_lstm_smote.ipynb      # LSTM + SMOTE (Synthetic Minority Over-sampling)
│   ├── 07_indobert_lora.ipynb   # IndoBERTweet-LoRA Vanilla Adapter
│   ├── 08_tapt_indobert_lora.ipynb # TAPT IndoBERTweet-LoRA (MLM 3 Epoch + Downstream)
│   └── 09_summary_model.ipynb   # Master Summary Table Dinamis & Grafik 4-Panel
├── Data/
│   ├── README.md                # Dokumentasi detail Data v1 vs Data v2
│   ├── data_preprocessed_with_emoticon.csv # [DATA V1] Data hasil pra-pemrosesan awal
│   ├── raw/banjir.csv           # Data mentah scraping Twitter (8.650 baris)
│   ├── interim/                 # Data transisi audit & rekonstruksi LLM
│   ├── processed/
│   │   └── data_clean_final.csv # [DATA V2] DATA UTAMA BERSIH (8.648 baris)
│   └── simulated/               # Dataset skenario simulasi (111, 631, 811)
├── kamus/
│   └── colloquial-indonesian-lexicon.csv # Kamus 4.334 leksikon slang bahasa Indonesia
├── Output/
│   ├── saved_models/            # File model tersimpan (Pickle & LoRA Adapter)
│   ├── metrics/                 # JSON metrik master (experiment_metrics_summary.json)
│   ├── predictions/             # Prediksi CSV baris-demi-baris data uji
│   └── summary/                 # CSV komparasi dan visualisasi plot
├── SUMMARY_MODEL.md             # Dokumen ringkasan master metrik & analisis Bab IV
├── requirements.txt             # Dependensi Python
└── README.md                    # Halaman depan dokumentasi repositori
```

---

## 🎯 3. Master Comparison Table Hasil Akhir (Data Latih vs Data Uji Terkunci $n=1.730$)

Seluruh metrik berikut dimuat secara dinamis dari [`Output/metrics/experiment_metrics_summary.json`](Output/metrics/experiment_metrics_summary.json):

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Generalization Gap Acc (%) | Recall Netral (%) |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | 61.65% | 61.69% | **61.56%** | +10.40% | 28.81% |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | 64.88% | 64.35% | **64.60%** | +16.81% | 40.40% |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | 64.38% | 65.62% | **64.89%** | +22.40% | 48.68% |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | 57.34% | 56.95% | **52.72%** | +17.53% | 56.62% |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | 61.05% | 57.07% | **57.37%** | +0.63% | 43.71% |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | 73.18% | 74.88% | **73.90%** | +6.17% | 61.26% |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | 75.83% | 73.83% | **74.61%** | +6.34% | 52.65% |

---

## 🔬 4. Temuan Ilmiah Utama & Pembahasan Naskah Bab IV Tesis

### A. Mengapa Akurasi LSTM Baseline (70,92%) Lebih Tinggi dari RUS (53,06%) dan SMOTE (64,39%)?
1. **The Accuracy Paradox (Ilusi Akurasi)**:
   - Pada korpus tweet banjir ini, kelas mayoritas (Positif & Negatif) mencakup $\approx 82\%$ data uji. Model baseline yang memprediksi mayoritas secara alami akan mencetak akurasi global tinggi, namun menderita **Majority Collapse** (Recall Netral hanya **28,81%**).
2. **Information Loss pada RUS**:
   - Pemangkasan lebih dari 45% data mayoritas membuang ribuan variasi kata penting dalam konteks banjir lokal, menyebabkan akurasi anjlok drastis ke **53,06%**.
3. **Discrete Token Corruption pada SMOTE**:
   - Interpolasi linier SMOTE yang didesain untuk fitur kontinu menghasilkan *pseudo-tokens* acak tanpa arti semantik pada ruang integer kata diskrit, merusak mekanisme gerbang memori LSTM ke **64,39%**.
4. **Penyelamatan Nyata Kelas Minoritas**:
   - Teknik balancing berhasil menyelamatkan kelas minoritas Netral: Recall Netral melonjak dari **28,81% (Baseline)** ke **40,40% (Class Weight)** dan **48,68% (ROS)**.

### B. Superioritas Transformer vs LSTM
* Model berbasis Transformer IndoBERTweet mengungguli varian LSTM terbaik dengan lonjakan akurasi signifikan (**+9,14%** dari 70,92% ke 80,06%) dan lonjakan Macro F1 **+13,05%** (dari 61,56% ke 74,61%).
* Mekanisme *Bidirectional Self-Attention* mampu mengurai ungkapan sarkasme, singkatan bahasa gaul, dan struktur kalimat informal Twitter tanpa kehilangan informasi konteks panjang.

### C. Efektivitas Task-Adaptive Pretraining (TAPT)
* Pretraining lanjutan tanpa supervisi (*Masked Language Modeling* 3 epoch) pada korpus banjir berhasil mengadaptasi model pada istilah teknis kebencanaan lokal Sumatra, mendorong akurasi menembus batas psikologis 80% (**80,06%**).

---

## 💻 5. Panduan Menjalankan di Workstation Lokal (CPU)

Notebook `01` s/d `06` serta `09` dirancang sangat efisien dan dapat dijalankan di CPU laptop biasa (~1-3 menit per notebook):

### Langkah 1: Kloning Repositori
```bash
git clone -b main https://github.com/Emanxaa/sentiment_nidi.git
cd sentiment_nidi
```

### Langkah 2: Setup Lingkungan Python Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### Langkah 3: Buka Jupyter Lab / Notebook
```bash
jupyter lab
```

### Langkah 4: Urutan Eksekusi
1. Buka folder `notebooks/`.
2. Jalankan `01_data_processing.ipynb` (verifikasi data clean).
3. Jalankan model LSTM berurutan: `02_lstm_imbalance.ipynb`, `03_lstm_class_weight.ipynb`, `04_lstm_oversampling.ipynb`, `05_lstm_undersampling.ipynb`, `06_lstm_smote.ipynb`.
4. Jalankan `09_summary_model.ipynb`: Seluruh metrik model akan terkompilasi otomatis ke dalam tabel master dan grafik Seaborn 4-panel!

---

## ☁️ 6. Panduan Menjalankan di Kaggle (Cloud GPU)

Khusus untuk notebook Transformer (`07_indobert_lora.ipynb` dan `08_tapt_indobert_lora.ipynb`), eksekusi sangat disarankan menggunakan **Kaggle GPU gratis (Tesla T4)**:

### Langkah 1: Siapkan Dataset di Kaggle (Hanya 1 Kali)
1. Login ke [Kaggle](https://www.kaggle.com/) -> klik menu **Datasets** -> **+ New Dataset**.
2. Beri nama: `dataset-thesis-sentiment`.
3. Upload 2 file dari repositori lokal Anda:
   - `Data/processed/data_clean_final.csv`
   - `kamus/colloquial-indonesian-lexicon.csv`
4. Klik **Create**.

### Langkah 2: Buat / Import Notebook di Kaggle
1. Di Kaggle, klik menu **Code** -> **+ New Notebook**.
2. Klik menu **File** -> **Import Notebook** -> Pilih berkas `07_indobert_lora.ipynb` atau `08_tapt_indobert_lora.ipynb` dari laptop Anda.

### Langkah 3: Atur Konfigurasi Sesi Kaggle (PENTING)
Di panel sebelah kanan (**Notebook options / Settings**):
1. **Accelerator**: Pilih **GPU T4 x2** atau **GPU P100**.
2. **Internet**: Geser ke **ON** *(wajib agar library Transformers dapat mendownload base model dari HuggingFace)*.
3. **Input**: Klik **+ Add Input**, cari dataset yang dibuat di Langkah 1 (`dataset-thesis-sentiment`), lalu klik tanda `+` (Add).

> [!TIP]
> Kode di seluruh notebook memiliki fungsi `resolve_path()` yang **secara otomatis mendeteksi folder `/kaggle/input/`**, sehingga Anda tidak perlu mengubah path file apa pun!

### Langkah 4: Eksekusi & Simpan Hasil
* **Interaktif**: Klik **Run All** di toolbar atas (selesai ~6-8 menit).
* **Background Run**: Klik tombol **Save Version** di pojok kanan atas -> pilih **Save & Run All (Commit)** -> klik **Save**.

### Langkah 5: Unduh Hasil dari Kaggle
* Setelah selesai, masuk ke tab **Output** di panel kanan Kaggle.
* Unduh folder `Output/saved_models/` (bobot adapter) dan file `Output/metrics/experiment_metrics_summary.json`.
* Pindahkan file JSON hasil ke folder lokal `Output/metrics/`, lalu jalankan `09_summary_model.ipynb` di laptop untuk langsung menampilkan ringkasan terbaru!

---

## 💾 7. Akses File Model Tersimpan di Python

Semua model LSTM disimpan via pickle, dan IndoBERT disimpan via adapter LoRA:

```python
import pickle
import pandas as pd

# 1. Memuat Model LSTM
with open("Output/saved_models/model_lstm_02_baseline_.pkl", "rb") as f:
    model_lstm = pickle.load(f)
with open("Output/saved_models/tokenizer_lstm_02_baseline_.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 2. Memuat Ringkasan Metrik
df_summary = pd.read_csv("Output/predictions/tabel_komparasi_seluruh_model.csv")
print(df_summary[["Model", "Train Accuracy (%)", "Test Accuracy (%)", "Macro F1 (%)", "Recall Netral (%)"]])
```
