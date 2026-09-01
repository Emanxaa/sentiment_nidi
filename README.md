# Thesis-LSTM-IndoBERT: Analisis Sentimen & Topik Tweet Banjir

Implementasi penelitian tesis untuk **klasifikasi sentimen 3-kelas (*negatif*, *netral*, *positif*)** pada tweet bencana banjir di Sumatera menggunakan pendekatan **IndoBERTweet-LoRA**, **BiLSTM**, dan **LSTM**, serta **Topic Modeling** berbasis **BERTopic**, **TF-IDF**, dan **LDA**.

---

## 🎯 Model & Hasil Utama

Model terbaik yang dikunci untuk pelaporan tesis:
- **Arsitektur**: `indolem/indobertweet-base-uncased` + PEFT LoRA ($r=16, \alpha=32$)
- **Data Input**: Kolom kanonik `text_bert` (*label corrected*)
- **Decision Layer**: *Threshold Calibration* ($w=[1, 1.5, 1]$)

| Metrik (Test, $n=1730$) | Baseline Awal | Final (LoRA + Kalibrasi) |
|---|---|---|
| **Accuracy** | 0.7954 | **0.7746** |
| **Macro F1** | 0.7328 | **0.7394** |
| **Netral Recall** | 0.5000 | **0.6690 (+0.17)** |
| **Netral F1** | 0.5500 | **0.6010** |

---

## 🚀 Quick Start

### 1. Instalasi Dependensi
```bash
git clone https://github.com/emanuelembuaijdak/Thesis-LSTM-IndoBERT.git
cd Thesis-LSTM-IndoBERT
pip install -r requirements.txt
```

### 2. Pembangkitan Notebook Eksperimen
Setiap eksperimen didefinisikan secara deklaratif di `configs/`. Gunakan generator untuk membuat notebook mandiri:
```bash
python tools/generate_notebook.py --config configs/exp_p1_weightedce.yaml
```

### 3. Verifikasi Metrik & Deteksi Collapse
Evaluasi prediksi model secara independen dan lakukan uji signifikansi:
```bash
python -m quality_pipeline.verify_metrics --preds hasil_prediksi_test.csv --y-true label_aktual --y-pred label_prediksi
```

### 4. Eksekusi Cloud via Kaggle CLI
```bash
kaggle kernels push -p temp_kernel/exp_p1_weightedce
kaggle kernels status emanuelembuaijdak/thesis-lora-p1-weightedce
kaggle kernels output emanuelembuaijdak/thesis-lora-p1-weightedce -p .kaggle-outputs/
```

---

## 📚 Indeks Dokumentasi

| Dokumen | Deskripsi |
|---|---|
| [TASK_BOARD.md](docs/TASK_BOARD.md) | Roadmap backlog teknis (P0, P1, P2) |
| [LAPORAN_AKHIR_EKSPERIMEN.md](docs/LAPORAN_AKHIR_EKSPERIMEN.md) | Laporan hasil eksperimen Bab IV tesis & tabel metrik |
| [HANDOVER.md](docs/HANDOVER.md) | Panduan serah terima proyek, evolusi kerja, & deliverables |
| [MODELS.md](docs/MODELS.md) | Spesifikasi teknis arsitektur LSTM, BiLSTM, IndoBERTweet-LoRA, & Topic Modeling |
| [P0_VERIFIKASI_EVALUASI.md](docs/P0_VERIFIKASI_EVALUASI.md) | Analisis diagnosis root cause metrik lama vs kanonik |
| [DATA_FLOW.md](docs/DATA_FLOW.md) | Diagram alur end-to-end data dan split dataset |
| [GUIDE_KAGGLE_CLI.md](docs/GUIDE_KAGGLE_CLI.md) | Panduan operasional pelatihan GPU via Kaggle CLI |
| [GUIDE_DATA_QUALITY_PIPELINE.md](docs/GUIDE_DATA_QUALITY_PIPELINE.md) | Panduan menjalankan pipeline 6 fase kualitas data |

---

## ⚖️ Aturan & Integritas Repositori

Aturan operasional dan tata kelola reproduksibilitas dapat dibaca pada [AGENTS.MD](AGENTS.MD).
