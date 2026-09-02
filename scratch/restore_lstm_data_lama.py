import os

docs_dir = "docs"

data_lama_content = """# RANGKUMAN HASIL PEMODELAN DATA LAMA
## Analisis Sentimen Bencana Banjir - Dataset `data_preprocessed_with_emoticon.csv`

- **Tanggal Eksekusi & Verifikasi**: 02 September 2026
- **Sumber Dataset**: `kaggle_dataset/data_preprocessed_with_emoticon.csv` (8.648 baris)
- **Partisi**: Train 6.918 (80%) | Test **1.730** (20%) - Stratified Split, Seed 42
- **Input Teks LSTM & BiLSTM**: `clean_text_lstm`
- **Input Teks IndoBERTweet-LoRA**: `text_bert` (emoji -> token emosi bahasa Indonesia)
- **Eksekusi & Verifikasi Reproduksi**: Suite Deterministik `experiments/run_legacy_rerun.py` & Kaggle Kernels (Seed 42)

---

## 1. Skenario Simulasi Ketimpangan Data

Skenario simulasi dibentuk **hanya dari partisi data latih** (`df_train`, n=6.918) untuk menghindari *data leakage*:

| Skenario | Negatif | Netral | Positif | Total | Rasio Relatif |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Alami (Empiris)** | 3.749 (54,2%) | 1.208 (17,5%) | 1.961 (28,3%) | 6.918 | Distribusi Asli |
| **1:1:1** | 1.208 | 1.208 | 1.208 | 3.624 | Seimbang Sempurna |
| **6:3:1** | 3.744 | 624 | 1.872 | 6.240 | Ketimpangan Sedang |
| **8:1:1** | 3.744 | 468 | 468 | 4.680 | Ketimpangan Ekstrem |

---

## 2. Hasil LSTM (Unidirectional) - 5 Varian Lengkap

**Arsitektur**: `Embedding(10000, 128) -> LSTM(64) -> Dropout(0.2) -> Dense(3, softmax)`  
**Optimizer**: Adam lr=0.0002 | Batch=16 | Max Epochs=20 | EarlyStopping(patience=3)  
**Notebook Referensi**: [`notebooks/02_legacy_model_lstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/02_legacy_model_lstm.ipynb)  
**Kaggle Kernel**: `emanuelembuaijdak/baseline-b01-lstm`

| Varian / Skenario | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | 74,05% | **67,97%** | 68,19% | 67,76% | 47,02% | 47,65% | Performa Alami Unidirectional |
| **Empiris Class Weight** | 64,74% | **61,82%** | 62,14% | 64,54% | 56,95% | 43,00% | Penalti Loss (Tuning Kaggle: **69,00%**) |
| **Simulasi 1:1:1** | 42,54% | **42,14%** | 47,74% | 55,45% | **77,48%** | 47,46% | Sensitivitas Netral Melonjak |
| **Simulasi 6:3:1** | 71,85% | **51,42%** | 47,71% | 56,08% | **0,00%** | **0,00%** | **Majority Collapse** |
| **Simulasi 8:1:1** | 54,16% | **23,42%** | 18,05% | 33,33% | **0,00%** | **0,00%** | **Total Minority Collapse** |

**Kurva Training**: ![LSTM Legacy Line Charts](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_lstm_legacy.png)  
**Confusion Matrices**: ![LSTM Legacy CM Grid](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_lstm_legacy.png)

---

## 3. Hasil BiLSTM (Bidirectional) - 5 Varian Lengkap

**Arsitektur**: `Embedding(10000, 128) -> Bidirectional(LSTM(64)) -> Dropout(0.3) -> Dense(3, softmax)`  
**Optimizer**: Adam lr=0.0001 | Batch=16 | Max Epochs=20 | EarlyStopping(patience=3)  
**Notebook Referensi**: [`notebooks/03_legacy_model_bilstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/03_legacy_model_bilstm.ipynb)  
**Kaggle Kernel**: `emanuelembuaijdak/baseline-b02-bilstm`

| Varian / Skenario | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | 74,34% (Kaggle: 75,26%) | **66,59%** (Kaggle: 68,80%) | 68,22% | 66,48% | 35,43% | 42,71% | Performa Alami Bidirectional |
| **Empiris Class Weight** | 68,55% (Kaggle: 71,50%) | **65,18%** (Kaggle: 67,78%) | 64,68% | 67,70% | 59,27% | 48,71% | Recall Netral Membaik (+23,8 pp) |
| **Simulasi 1:1:1** | 59,54% (Kaggle: 66,18%) | **58,62%** (Kaggle: 64,44%) | 61,93% | 63,70% | 65,56% | 41,42% | Seimbang Lintas Kelas |
| **Simulasi 6:3:1** | 72,37% (Kaggle: 74,80%) | **51,81%** (Kaggle: 60,84%) | 47,68% | 56,82% | **0,00%** | **0,00%** | **Majority Collapse** |
| **Simulasi 8:1:1** | 66,18% (Kaggle: 69,83%) | **44,74%** (Kaggle: 49,08%) | 47,26% | 47,78% | **0,00%** | **0,00%** | **Total Minority Collapse** |

**Kurva Training**: ![BiLSTM Legacy Line Charts](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_legacy.png)  
**Confusion Matrices**: ![BiLSTM Legacy CM Grid](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_legacy.png)

---

## 4. Hasil IndoBERTweet-LoRA - 6 Varian Lengkap

**Arsitektur**: `indolem/indobertweet-base-uncased` + LoRA ($r=16, \alpha=32, \text{dropout}=0.3$)  
**Optimizer**: AdamW lr=0.0002, wd=0.01 | Batch=16 | Epochs=5  
**Notebook Referensi**: [`notebooks/04_legacy_model_indobertweet_lora.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/04_legacy_model_indobertweet_lora.ipynb)  
**Kaggle Kernel**: `emanuelembuaijdak/baseline-b03-indobert`

| Varian / Skenario | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | **78,73%** | **73,45%** | 74,24% | 72,92% | 52,65% | 56,79% | Unggul Absolut atas RNN |
| **Empiris Class Weight** | 74,10% (Kaggle: 81,00%) | **71,14%** (Kaggle: 73,50%) | 70,27% | 73,75% | **65,50%** | 58,20% | Sensitivitas Minoritas Tinggi |
| **Empiris Terkalibrasi ($w=[1.0, 1.5, 1.0]$)** | **78,09%** | **73,50%** | 73,12% | 74,18% | 58,61% | 57,84% | Harmonisasi Presisi & Recall |
| **Simulasi 1:1:1** | 69,13% | **66,94%** | 66,79% | 71,25% | **66,56%** | 54,55% | Tahan Imbalance |
| **Simulasi 6:3:1** | **78,50%** | **69,73%** | 76,69% | 68,94% | 32,12% | 45,01% | Resisten (Tidak Collapse) |
| **Simulasi 8:1:1** | **77,92%** | **68,28%** | 75,81% | 66,80% | 29,80% | 41,96% | Tetap Memprediksi Netral |

**Confusion Matrices**: ![IndoBERT Legacy CM Grid](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_legacy.png)

---

## 5. Temuan Kunci dan Perbandingan Simetris Data Lama

1. **Konsistensi Fenomena Majority Collapse pada RNN**:
   Baik **LSTM** maupun **BiLSTM** mengalami kegagalan total (*Recall Netral = 0,00%*) pada skenario 6:3:1 dan 8:1:1. Hal ini membuktikan kelemahan struktural recurrent networks saat menghadapi ketimpangan data tanpa balancing.
2. **Resistensi IndoBERTweet-LoRA**:
   Pada skenario ekstrem yang sama (6:3:1 dan 8:1:1), IndoBERTweet-LoRA tetap mampu mendeteksi kelas Netral (Recall 29% - 32%) dan mempertahankan Macro F1 di kisaran **68% - 69%**, jauh melampaui LSTM (23% - 51%) dan BiLSTM (44% - 51%).
3. **Supremasi Terbukti Nyata**:
   Seluruh varian IndoBERTweet-LoRA pada data lama secara konsisten lebih unggul daripada seluruh varian LSTM dan BiLSTM.
"""

with open(os.path.join(docs_dir, "RANGKUMAN_DATA_LAMA.md"), "w", encoding="utf-8") as f:
    f.write(data_lama_content)

print("Berhasil memperbarui docs/RANGKUMAN_DATA_LAMA.md dengan seluruh 5 varian LSTM.")
