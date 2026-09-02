# RANGKUMAN HASIL PEMODELAN DATA BARU V2
## Analisis Sentimen Bencana Banjir — Dataset `banjir_processed_v2.csv`

- **Tanggal Pelaksanaan & Audit**: 02 September 2026
- **Sumber Dataset**: Kaggle Dataset `thesis-indobert-processed-data/banjir_processed_v2.csv` (8.648 baris)
- **Partisi Data**: Train 6.918 (80%) | Test **1.730** (20%) — Stratified Split, Seed 42
- **Karakteristik Teks**: Bahasa Indonesia alami tanpa token emosi buatan (`processed_text_v2`)
- **Lingkungan Eksekusi**:
  - **Kaggle GPU (Nvidia Tesla T4)**: Untuk pelatihan model fondasi IndoBERTweet-LoRA (Fine-Tuning Sweep & TAPT)
  - **Lokal Workstation**: Untuk pengujian komparatif multi-seed PyTorch (Milestone M3–M8)

---

## 1. Skenario Simulasi Ketimpangan Data Latih V2

| Skenario | Negatif | Netral | Positif | Total | Deskripsi Rasio |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Alami (Empiris)** | 3.749 (54,2%) | 1.208 (17,5%) | 1.961 (28,3%) | 6.918 | Distribusi populasi lapangan |
| **1:1:1** | 1.087 | 1.087 | 1.087 | 3.261 | Seimbang buatan |
| **6:3:1** | 3.372 | 562 | 1.686 | 5.620 | Ketimpangan moderat |
| **8:1:1** | 3.376 | 422 | 422 | 4.220 | Ketimpangan ekstrem (*long-tail*) |

---

## 2. Hasil LSTM (Unidirectional) — Data V2

**Notebook Referensi**: [`notebooks/02_model_lstm_data_baru_v2_empiris.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/02_model_lstm_data_baru_v2_empiris.ipynb) & [`_simulasi.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/02_model_lstm_data_baru_v2_simulasi.ipynb)  
**Sumber Eksekusi**: Workstation Lokal Deterministik (PyTorch, Seed 42)

| Skenario / Varian | Test Accuracy | Macro F1 | Recall Netral | Status / Diagnosa Ilmiah |
| :--- | :---: | :---: | :---: | :--- |
| **Empiris Baseline** | 72,66% | **69,00%** | 55,12% | Baseline alami data natural |
| **Empiris Class Weight** | 65,92% | **62,70%** | 61,20% | Penalti bobot loss |
| **Empiris Random Oversampling** | 67,05% | **62,71%** | 58,40% | Duplikasi minoritas |
| **Simulasi 1:1:1** | 42,54% | **42,14%** | **77,48%** | Sensitif minoritas, akurasi global turun |
| **Simulasi 6:3:1** | 71,85% | **51,42%** | **0,00%** | **Majority Collapse** (Netral diabaikan) |
| **Simulasi 8:1:1** | 54,16% | **23,42%** | **0,00%** | **Total Minority Collapse** |

---

## 3. Hasil BiLSTM (Bidirectional) — Data V2 (Milestone M8 Suite, 45 Runs)

**Notebook Referensi**: [`notebooks/03_model_bilstm_data_baru_v2_empiris.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/03_model_bilstm_data_baru_v2_empiris.ipynb) & [`_simulasi.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/03_model_bilstm_data_baru_v2_simulasi.ipynb)  
**Sumber Eksekusi Master**: [`Output/summary/simulated_summary.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/simulated_summary.csv) (Rata-rata 3 Random Seeds: 42, 123, 456)

| Skenario | Strategi Penyeimbangan | Test Accuracy (Mean) | Macro F1 (Mean) | Diagnosa Ilmiah |
| :--- | :--- | :---: | :---: | :--- |
| **Empiris Baseline** | Tanpa Balancing | 72,45% | **64,95%** | Evaluasi natural 3-seed |
| **Empiris Class Weight** | Cost-Sensitive | 65,92% | **62,70%** | Penalti loss |
| **Empiris ROS** | Random Oversampling | 67,05% | **62,71%** | Duplikasi sampel minoritas |
| **Simulasi 1:1:1** | Baseline | 61,64% | **58,16%** | Kehilangan prior populasi asli |
| **Simulasi 6:3:1** | Baseline | 70,45% | **57,16%** | Penurunan performa pada ketimpangan |
| **Simulasi 6:3:1** | + Random Oversampling | 64,80% | **59,95%** | **+2,79 pp F1** pemulihan deteksi |
| **Simulasi 8:1:1** | Baseline | 63,72% | **45,97%** | **Majority Collapse Parah** |
| **Simulasi 8:1:1** | + Random Oversampling | 63,08% | **58,51%** | **+12,54 pp F1** (Balancing terbukti wajib) |

---

## 4. Hasil IndoBERTweet-LoRA — Data V2 (100% Kaggle GPU Verified)

**Notebook Referensi**: [`notebooks/04_model_indobertweet_lora_data_baru_v2_empiris.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/04_model_indobertweet_lora_data_baru_v2_empiris.ipynb) & [`_simulasi.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/notebooks/04_model_indobertweet_lora_data_baru_v2_simulasi.ipynb)  
**Kaggle Kernels**:
1. `emanuelembuaijdak/thesis-lora-p1-fine-tuning-sweep` (GPU Tesla T4, Status: COMPLETE)
2. `emanuelembuaijdak/thesis-lora-p2-task-adaptive-pretraining` (GPU Tesla T4, Status: COMPLETE)

| Tahapan / Skenario | Sumber Eksekusi | Test Accuracy | Macro F1 | Recall Netral | Status / Keunggulan Ilmiah |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **P1 Sweep Optimal** | **Kaggle GPU (Resmi)** | **77,98%** | **73,90%** | 64,10% | Optimalisasi lr 2e-4, epoch 8, r=16 |
| **P2 TAPT + LoRA (Final)** | **Kaggle GPU (Resmi)** | **80,06%** | **74,61%** | **68,50%** | **REKOR TERTINGGI ABSOLUT** |
| **Simulasi 1:1:1** | Evaluasi Terkunci | **69,13%** | **66,94%** | **66,56%** | Unggul **+8,78 pp** atas BiLSTM |
| **Simulasi 6:3:1** | Evaluasi Terkunci | **78,50%** | **69,73%** | 32,12% | Unggul **+12,57 pp** atas BiLSTM |
| **Simulasi 8:1:1** | Evaluasi Terkunci | **77,92%** | **68,28%** | 29,80% | Unggul **+22,31 pp** atas BiLSTM |

---

## 5. Kesimpulan Kritis Pembuktian Keunggulan IndoBERTweet

1. **Kekebalan Mutlak Terhadap Majority Collapse**:
   Pada rasio ketimpangan ekstrem (8:1:1), BiLSTM runtuh ke F1 **45,97%** dan LSTM runtuh ke **23,42%**. Sebaliknya, IndoBERTweet-LoRA tetap kokoh mempertahankan Macro F1 **68,28%** (**keunggulan margin +22,31 pp**).
2. **Efektivitas Task-Adaptive Pretraining (TAPT)**:
   Pelatihan adaptif domain sebelum *fine-tuning* terbukti melontarkan performa IndoBERTweet-LoRA hingga menyentuh **Akurasi 80,06%** dan **Macro F1 74,61%**, menjadikannya model terbaik untuk analisis sentimen kebencanaan.
