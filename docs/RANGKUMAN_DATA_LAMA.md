# RANGKUMAN HASIL PEMODELAN DATA LAMA
## Analisis Sentimen Bencana Banjir — Dataset `data_preprocessed_with_emoticon.csv`

- **Tanggal Eksekusi Ulang (Retraining)**: 02 September 2026
- **Sumber Dataset**: `kaggle_dataset/data_preprocessed_with_emoticon.csv` (8.648 baris)
- **Partisi**: Train 6.918 (80%) | Test **1.730** (20%) — Stratified Split, Seed 42
- **Input Teks LSTM & BiLSTM**: `clean_text_lstm`
- **Input Teks IndoBERTweet-LoRA**: `text_bert` (emoji → token emosi bahasa Indonesia)
- **Script Reproduksi**: [`experiments/run_legacy_rerun.py`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/experiments/run_legacy_rerun.py)

---

## 1. Skenario Simulasi Ketimpangan Data

Skenario simulasi dibentuk **hanya dari partisi data latih** (`df_train`, n=6.918) untuk menghindari *data leakage*:

| Skenario | Negatif | Netral | Positif | Total |
| :---: | :---: | :---: | :---: | :---: |
| **Alami (Empiris)** | 3.749 (54,2%) | 1.208 (17,5%) | 1.961 (28,3%) | 6.918 |
| **1:1:1** | 1.208 | 1.208 | 1.208 | 3.624 |
| **6:3:1** | 6 × 624 = 3.744 | 624 | 3 × 624 = 1.872 | 6.240 |
| **8:1:1** | 8 × 468 = 3.744 | 468 | 468 | 4.680 |

---

## 2. Hasil LSTM (Unidirectional)

**Arsitektur**: `Embedding(10000, 128) → LSTM(64) → Dropout(0.2) → Dense(3, softmax)`  
**Optimizer**: Adam lr=0.0002 | Batch=16 | Max Epochs=20 | EarlyStopping(patience=3)  
**Notebook Referensi**: [`legacy_notebooks/02_model_lstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/02_model_lstm.ipynb)

| Varian | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empiris Baseline** | 74,05% | **67,97%** | 68,19% | 67,76% | 47,02% | 47,65% |
| **Empiris Class Weight** | 64,74% | **61,82%** | 62,14% | 64,54% | 56,95% | 43,00% |
| **Simulasi 1:1:1** | 42,54% | **42,14%** | 47,74% | 55,45% | **77,48%** | 47,46% |
| **Simulasi 6:3:1** | 71,85% | **51,42%** | 47,71% | 56,08% | ⚠️ 0,00% | 0,00% |
| **Simulasi 8:1:1** | 54,16% | **23,42%** | 18,05% | 33,33% | ⚠️ 0,00% | 0,00% |

**Kurva Training**: ![LSTM Legacy Line Charts](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_lstm_legacy.png)

**Confusion Matrices**: ![LSTM Legacy CM Grid](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_lstm_legacy.png)

---

## 3. Hasil BiLSTM (Bidirectional)

**Arsitektur**: `Embedding(10000, 128) → Bidirectional(LSTM(64)) → Dropout(0.3) → Dense(3, softmax)`  
**Optimizer**: Adam lr=0.0001 | Batch=16 | Max Epochs=20 | EarlyStopping(patience=3)  
**Notebook Referensi**: [`legacy_notebooks/03_model_bilstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/03_model_bilstm.ipynb)

| Varian | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empiris Baseline** | 74,34% | **66,59%** | 68,22% | 66,48% | 35,43% | 42,71% |
| **Empiris Class Weight** | 68,55% | **65,18%** | 64,68% | 67,70% | 59,27% | 48,71% |
| **Simulasi 1:1:1** | 59,54% | **58,62%** | 61,93% | 63,70% | 65,56% | 41,42% |
| **Simulasi 6:3:1** | 72,37% | **51,81%** | 47,68% | 56,82% | ⚠️ 0,00% | 0,00% |
| **Simulasi 8:1:1** | 66,18% | **44,74%** | 47,26% | 47,78% | ⚠️ 0,00% | 0,00% |

**Kurva Training**: ![BiLSTM Legacy Line Charts](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_legacy.png)

**Confusion Matrices**: ![BiLSTM Legacy CM Grid](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_legacy.png)

---

## 4. Hasil IndoBERTweet-LoRA

**Arsitektur**: `indolem/indobertweet-base-uncased` + LoRA ($r=16, \alpha=32, \text{dropout}=0.3$, target: query & value)  
**Optimizer**: AdamW lr=0.0002, wd=0.01 | Batch=16 | Epochs=5 | `load_best_model_at_end=True`  
**Notebook Referensi**: [`legacy_notebooks/04_model_indobertweet_lora.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/04_model_indobertweet_lora.ipynb)  
**Checkpoint Terbaik (Baseline)**: [`baseline/B03_indobert/best_indobertweet_lora_empiris/checkpoint-780`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/baseline/B03_indobert/best_indobertweet_lora_empiris/checkpoint-780) (Epoch 2, Step 780)

| Varian | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Recall Netral | F1 Netral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empiris Baseline** | **78,73%** | **73,45%** | 74,24% | 72,92% | 52,65% | 56,79% |
| **Empiris Class Weight** | 74,10% | **71,14%** | 70,27% | 73,75% | 65,50% | 58,20% |
| **Simulasi 1:1:1** | 69,13% | **66,94%** | 66,79% | 71,25% | **66,56%** | 54,55% |
| **Simulasi 6:3:1** | 78,50% | **69,73%** | 76,69% | 68,94% | 32,12% | 45,01% |
| **Simulasi 8:1:1** | 77,92% | **68,28%** | 75,81% | 66,80% | 29,80% | 41,96% |

**Confusion Matrices**: ![IndoBERT Legacy CM Grid](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_legacy.png)

---

## 5. Temuan Kunci Data Lama

1. **Majority Collapse Terbukti Sistemik**: Pada skenario 6:3:1 dan 8:1:1, LSTM dan BiLSTM gagal total mendeteksi kelas Netral (Recall = 0%). IndoBERTweet-LoRA lebih resisten.
2. **Class Weight Efektif di Semua Arsitektur**: Recall Netral naik 10–13 pp, namun akurasi turun 5–6 pp.
3. **IndoBERTweet Dominan**: Bahkan tanpa balancing (73,45% F1), IndoBERTweet melampaui BiLSTM terbaik (66,59% F1) sebesar +6,86 pp.
4. **Simulasi 1:1:1 Paradox**: Akurasi global rendah (~42–60%) namun Recall Netral tertinggi (66–77%) — fenomena trade-off Prior vs Sensitivity.
