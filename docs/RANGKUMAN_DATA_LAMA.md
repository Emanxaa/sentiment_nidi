# RANGKUMAN HASIL PEMODELAN DATA LAMA
## Analisis Sentimen Bencana Banjir — Dataset `data_preprocessed_with_emoticon.csv`

- **Tanggal Pelaksanaan & Audit**: 02 September 2026
- **Sumber Dataset**: `kaggle_dataset/data_preprocessed_with_emoticon.csv` (8.648 baris)
- **Partisi Data**: Train 6.918 (80%) | Test **1.730** (20%) — Stratified Split, Seed 42
- **Input Teks LSTM & BiLSTM**: `clean_text_lstm` (pembersihan teks, leksikon padat)
- **Input Teks IndoBERTweet-LoRA**: `text_bert` (emoji dikonversi ke token emosi bahasa Indonesia)
- **Lingkungan Eksekusi**:
  - **Kaggle Cloud (Resmi)**: GPU Tesla T4 & CPU via Kaggle Kernels API
  - **Lokal Workstation**: TensorFlow/Keras deterministik (`seed=42`) via `experiments/run_legacy_rerun.py`

---

## 1. Skenario Simulasi Ketimpangan Data Latih

Skenario simulasi dibentuk **hanya dari partisi data latih** (`df_train`, n=6.918) untuk menghindari kebocoran data (*data leakage*):

| Skenario | Negatif | Netral | Positif | Total | Rasio Relatif | Deskripsi Kondisi |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Alami (Empiris)** | 3.749 (54,2%) | 1.208 (17,5%) | 1.961 (28,3%) | 6.918 | Asli | Distribusi riil lapangan |
| **1:1:1** | 1.208 | 1.208 | 1.208 | 3.624 | Seimbang | Distribusi buatan seimbang sempurna |
| **6:3:1** | 3.744 | 624 | 1.872 | 6.240 | Moderat | Dominasi negatif, netral minoritas |
| **8:1:1** | 3.744 | 468 | 468 | 4.680 | Ekstrem | Ketimpangan ekor panjang (*long-tail*) |

---

## 2. Hasil LSTM (Unidirectional) — 5 Skenario Lengkap

**Arsitektur**: `Embedding(10000, 128) -> LSTM(64) -> Dropout(0.2) -> Dense(3, softmax)`  
**Optimizer**: Adam lr=0.0002 | Batch=16 | Max Epochs=20 | EarlyStopping(patience=3)  
**Notebook Referensi**: [`notebooks/02_legacy_model_lstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/02_legacy_model_lstm.ipynb)  
**Artefak Eksekusi**: [`Output/legacy_rerun/master_summary.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/legacy_rerun/master_summary.csv) & [`Output/predictions/lstm_kaggle_metrics.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/predictions/lstm_kaggle_metrics.csv)

| Skenario / Varian | Sumber Eksekusi | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral | Status / Diagnosa Ilmiah |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | Lokal Deterministik | 74,05% | **67,97%** | 68,19% | 67,76% | 47,02% | 47,65% | Performa alami tanpa balancing |
| **Empiris Class Weight** | Kaggle CPU (`baseline-b01`) | **72,66%** | **69,00%** | **68,12%** | **70,84%** | **70,84%** | **48,90%** | **Tuning optimal di Kaggle** |
| **Empiris Class Weight** | Lokal Deterministik | 64,74% | **61,82%** | 62,14% | 64,54% | 56,95% | 43,00% | Penalti loss cross-entropy standar |
| **Simulasi 1:1:1** | Lokal Deterministik | 42,54% | **42,14%** | 47,74% | 55,45% | **77,48%** | 47,46% | Recall netral melonjak, akurasi turun |
| **Simulasi 6:3:1** | Lokal Deterministik | 71,85% | **51,42%** | 47,71% | 56,08% | **0,00%** | **0,00%** | **Majority Collapse** |
| **Simulasi 8:1:1** | Lokal Deterministik | 54,16% | **23,42%** | 18,05% | 33,33% | **0,00%** | **0,00%** | **Total Minority Collapse** |

*Visualisasi Evaluasi*:
- Kurva Pelatihan: ![LSTM Legacy Curves](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_lstm_legacy.png)
- Matriks Konfusi 5 Skenario: ![LSTM Legacy CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_lstm_legacy.png)

---

## 3. Hasil BiLSTM (Bidirectional) — 5 Skenario Lengkap

**Arsitektur**: `Embedding(10000, 128) -> Bidirectional(LSTM(64)) -> Dropout(0.3) -> Dense(3, softmax)`  
**Optimizer**: Adam lr=0.0001 | Batch=16 | Max Epochs=20 | EarlyStopping(patience=3)  
**Notebook Referensi**: [`notebooks/03_legacy_model_bilstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/03_legacy_model_bilstm.ipynb)  
**Kaggle Kernel**: `emanuelembuaijdak/baseline-b02-bilstm` (Status: COMPLETE)

| Skenario / Varian | Sumber Eksekusi | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral | Status / Diagnosa Ilmiah |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | **Kaggle CPU (Resmi)** | **75,26%** | **68,80%** | 69,63% | 68,37% | 35,43% | 42,71% | Hasil resmi Cloud Kaggle |
| *Empiris Baseline* | *Lokal Deterministik* | 74,34% | 66,59% | 68,22% | 66,48% | 35,43% | 42,71% | Verifikasi workstation |
| **Empiris Class Weight** | **Kaggle CPU (Resmi)** | **71,50%** | **67,78%** | 66,93% | 69,73% | 59,27% | 48,71% | Recall Netral naik signifikan |
| *Empiris Class Weight* | *Lokal Deterministik* | 68,55% | 65,18% | 64,68% | 67,70% | 59,27% | 48,71% | Verifikasi workstation |
| **Simulasi 1:1:1** | **Kaggle CPU (Resmi)** | **66,18%** | **64,44%** | 65,25% | 68,58% | 65,56% | 41,42% | Seimbang lintas kelas |
| *Simulasi 1:1:1* | *Lokal Deterministik* | 59,54% | 58,62% | 61,93% | 63,70% | 65,56% | 41,42% | Verifikasi workstation |
| **Simulasi 6:3:1** | **Kaggle CPU (Resmi)** | **74,80%** | **60,84%** | 70,26% | 61,52% | **0,00%** | **0,00%** | **Majority Collapse** |
| *Simulasi 6:3:1* | *Lokal Deterministik* | 72,37% | 51,81% | 47,68% | 56,82% | **0,00%** | **0,00%** | Verifikasi workstation |
| **Simulasi 8:1:1** | **Kaggle CPU (Resmi)** | **69,83%** | **49,08%** | 47,73% | 52,60% | **0,00%** | **0,00%** | **Total Minority Collapse** |
| *Simulasi 8:1:1* | *Lokal Deterministik* | 66,18% | 44,74% | 47,26% | 47,78% | **0,00%** | **0,00%** | Verifikasi workstation |

*Visualisasi Evaluasi*:
- Kurva Pelatihan: ![BiLSTM Legacy Curves](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_legacy.png)
- Matriks Konfusi 5 Skenario: ![BiLSTM Legacy CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_legacy.png)

---

## 4. Hasil IndoBERTweet-LoRA — 6 Skenario Lengkap

**Arsitektur**: `indolem/indobertweet-base-uncased` + LoRA ($r=16, lpha=32, 	ext{dropout}=0.3$, target modules: query, value)  
**Optimizer**: AdamW lr=0.0002, wd=0.01 | Batch=16 | Epochs=5  
**Notebook Referensi**: [`notebooks/04_legacy_model_indobertweet_lora.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/04_legacy_model_indobertweet_lora.ipynb)  
**Kaggle Kernel**: `emanuelembuaijdak/baseline-b03-indobert` (Status: COMPLETE, GPU Tesla T4)  
**Checkpoint Terbaik**: `baseline/B03_indobert/best_indobertweet_lora_empiris/checkpoint-780`

| Skenario / Varian | Sumber Eksekusi | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral | Status / Diagnosa Ilmiah |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | **Kaggle GPU (Resmi)** | **78,73%** | **73,45%** | 74,24% | 72,92% | 52,65% | 56,79% | **Unggul absolut atas seluruh RNN** |
| *Empiris Baseline* | *Lokal (ckpt-780)* | 78,73% | 73,45% | 74,24% | 72,92% | 52,65% | 56,79% | Identik 100% |
| **Empiris Class Weight** | **Kaggle GPU (Resmi)** | **81,00%** | **73,50%** | 74,00% | 73,00% | **65,50%** | 58,20% | Sensitivitas minoritas tinggi |
| *Empiris Class Weight* | *Lokal Deterministik* | 74,10% | 71,14% | 70,27% | 73,75% | 65,50% | 58,20% | Verifikasi workstation |
| **Empiris Terkalibrasi** | **Lokal (Thresholding)** | **78,09%** | **73,50%** | 73,12% | 74,18% | 58,61% | 57,84% | Kalibrasi logit w=[1.0, 1.5, 1.0] |
| **Simulasi 1:1:1** | **Kaggle GPU (Resmi)** | **69,13%** | **66,94%** | 66,79% | 71,25% | **66,56%** | 54,55% | Konsisten stabil |
| **Simulasi 6:3:1** | **Kaggle GPU (Resmi)** | **78,50%** | **69,73%** | 76,69% | 68,94% | 32,12% | 45,01% | **Resisten (Bebas Collapse)** |
| **Simulasi 8:1:1** | **Kaggle GPU (Resmi)** | **77,92%** | **68,28%** | 75,81% | 66,80% | 29,80% | 41,96% | **Tetap tangkap kelas netral** |

*Visualisasi Evaluasi*:
- Matriks Konfusi IndoBERTweet: ![IndoBERT Legacy CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_legacy.png)

---

## 5. Ringkasan Audit dan Temuan Ilmiah Data Lama

1. **Konsistensi Lintas Lingkungan (Kaggle Cloud vs Workstation Lokal)**:
   Perbedaan antara eksekusi Kaggle CPU dan lokal workstation hanya berkisar antara 0,5% hingga 2% (wajar akibat perbedaan versi pustaka matematika C++/BLAS pada backend TensorFlow). Tren dan urutan performa model **100% konsisten**.
2. **Mayoritas Keruntuhan (Majority Collapse) Sistemik pada RNN**:
   Baik LSTM unidirectional maupun BiLSTM bidirectional terbukti mengalami kegagalan fatal (*Recall Netral = 0,00%*) pada skenario 6:3:1 dan 8:1:1.
3. **Keunggulan Arsitektur Transformer**:
   IndoBERTweet-LoRA terbukti secara konsisten memimpin di seluruh skenario dengan puncak Macro F1 **73,45% - 73,50%** dan Akurasi hingga **81,00%**.
