# Thesis-LSTM-IndoBERT: Analisis Sentimen & Topik Tweet Banjir

Implementasi penelitian tesis untuk **klasifikasi sentimen 3-kelas (*negatif*, *netral*, *positif*)** pada tweet bencana banjir di Sumatera menggunakan pendekatan **IndoBERTweet-LoRA**, **BiLSTM**, dan **LSTM**, serta **Topic Modeling** berbasis **BERTopic**, **TF-IDF**, dan **LDA**.

---

## 🎯 Model & Hasil Utama

Model terbaik yang dikunci untuk pelaporan tesis:
- **Arsitektur**: `indolem/indobertweet-base-uncased` + PEFT LoRA ($r=16, \alpha=32$)
- **Data Input**: Kolom kanonik `text_bert` (*label corrected & preprocessed v2*)
- **Decision Layer**: *Threshold Calibration* ($w=[1, 1.5, 1]$)

| Metrik (Test, $n=1730$) | Baseline LSTM | Baseline BiLSTM | Baseline IndoBERT | Final (LoRA + Kalibrasi) |
|---|:---:|:---:|:---:|:---:|
| **Accuracy** | 72.66% | 75.26% | 78.73% | **77.46%** |
| **Macro F1** | 0.6899 | 0.6880 | 0.7345 | **0.7394** |
| **Netral Recall** | 55.12% | 49.15% | 53.58% | **66.89% (+13.3%)** |
| **Netral F1** | 0.5283 | 0.5126 | 0.5638 | **0.6012 (Tembus $\ge 0.60$)** |
| **Cohen's Kappa ($\kappa$)** | 0.5694 (Sedang) | 0.5782 (Sedang) | 0.6387 (Kuat) | **0.6274 (Kuat)** |

---

## 📊 Dua Pilar Pencapaian Utama Proyek

### 1. Evaluasi Komparasi Model Baseline (Data Lama & Representasi Fitur)
Pengujian komparatif komprehensif pada data uji yang sama ($n = 1.730$) untuk membuktikan keunggulan representasi kontekstual Transformer dibanding RNN dan model linier klasik:
* **Transformer vs RNN (McNemar Test)**: IndoBERTweet-LoRA mengungguli LSTM secara signifikan ($\chi^2 = 38.42, p < 0.0001$).
* **Transformer vs Linear SVM**: Unggul signifikan ($\chi^2 = 46.18, p < 0.0001$).
* **Solusi Imbalance**: *Threshold Calibration* ($w=[1, 1.5, 1]$) menaikkan Recall Netral ke **66.89%** tanpa penurunan akurasi yang signifikan ($p = 0.3455$).
* Rincian tabel master: [`reports/benchmark_metrics.csv`](reports/benchmark_metrics.csv) dan [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](docs/LAPORAN_AKHIR_EKSPERIMEN.md).

### 2. Rekayasa Data Baru & Preprocessing Pipeline v2 (Task 01 — Task 08)
Pembangunan pipeline kualitas data terisolasi yang 100% reproduktif dari nol tanpa merusak integritas label:
* **Task 01 (Audit)**: Identifikasi 402 tweet terpotong UI scraping (`Data/interim/audit.csv`).
* **Task 02 (Conditional LLM Completion)**: 402/402 kalimat terpotong direkonstruksi utuh oleh LLM Gemini (`Data/interim/llm_completed.csv`).
* **Task 03 (Regex Refinement)**: 7.627 baris dibersihkan dari noise visual/URL/mention (`Data/interim/regex_clean.csv`).
* **Task 04 (Kamus Alay Normalization)**: 3.308 kata gaul dinormalisasi menggunakan 4.334 leksikon + *English Context Guard* (`Data/processed/banjir_processed_v2.csv`).
* **Task 05 (Dual Stream Preprocessing)**: Membentuk `text_bert` (Transformer) dan `clean_text_lstm` (RNN) di [`Data/processed/data_preprocessed_v2.csv`](Data/processed/data_preprocessed_v2.csv).
* **Task 06 (Stratified Split)**: 72% Train, 8% Val, 20% Test terkunci di [`Data/processed/split_data_v2.pkl`](Data/processed/split_data_v2.pkl).
* **Task 07–08 (Benchmark & Synthesis)**: Uji signifikansi statistik dan pembaruan tabel Bab IV [`experiments/results.csv`](experiments/results.csv).

---

## 🚀 Quick Start & Reproducibility

### 1. Instalasi Dependensi
```bash
git clone https://github.com/emanuelembuaijdak/Thesis-LSTM-IndoBERT.git
cd Thesis-LSTM-IndoBERT
pip install -r requirements.txt
```

### 2. Menjalankan Ulang Pipeline Data Baru (Task 01 s.d. 08)
```bash
# Task 01: Audit Data
python utils/data_audit.py

# Task 02: Rekonstruksi LLM Kalimat Terpotong
python utils/llm_completion.py

# Task 03: Pembersihan Regex
python utils/regex_refinement.py

# Task 04: Normalisasi Bahasa Gaul (Leksikon)
python utils/alay_normalization.py

# Task 05: Preprocessing Ganda (BERT & LSTM)
python utils/preprocess_pipeline.py

# Task 06: Pembagian Data Stratified
python utils/split_dataset.py

# Task 07: Benchmark Model Representasi
python utils/train_and_benchmark.py

# Task 08: Sintesis Akhir & Uji McNemar
python utils/evaluate_synthesis.py
```

---

## 📚 Indeks Dokumentasi

| Dokumen | Deskripsi |
|---|---|
| [DATA_FLOW.md](docs/DATA_FLOW.md) | Dokumentasi teknis lengkap pipeline data Task 01 s.d. Task 08 |
| [LAPORAN_AKHIR_EKSPERIMEN.md](docs/LAPORAN_AKHIR_EKSPERIMEN.md) | Laporan hasil eksperimen Bab IV tesis & tabel metrik master |
| [LOG_EKSPERIMEN.md](docs/LOG_EKSPERIMEN.md) | Log kronologis seluruh fase eksperimen & uji signifikansi |
| [TASK_BOARD.md](docs/TASK_BOARD.md) | Roadmap backlog teknis (P0, P1, P2) |
| [HANDOVER.md](docs/HANDOVER.md) | Panduan serah terima proyek, evolusi kerja, & deliverables |
| [MODELS.md](docs/MODELS.md) | Spesifikasi teknis arsitektur LSTM, BiLSTM, IndoBERTweet-LoRA |
| [AGENTS.md](AGENTS.md) | Standar operasional riset & tata kelola reproduksibilitas ilmiah |

---

## ⚖️ Aturan & Integritas Repositori

Aturan operasional dan tata kelola reproduksibilitas dapat dibaca pada [AGENTS.md](AGENTS.md).
