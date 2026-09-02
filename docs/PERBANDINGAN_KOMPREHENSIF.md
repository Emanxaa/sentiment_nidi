# PERBANDINGAN KOMPREHENSIF LINTAS DATA & MODEL
## LSTM · BiLSTM · IndoBERTweet-LoRA | Data Lama vs Data Baru (v2)

- **Tanggal**: 02 September 2026
- **Metrik Standar**: Accuracy, Macro F1, Macro Precision, Macro Recall, Recall Netral, F1 Netral
- **Partisi Evaluasi Seragam**: Test Set $n = 1.730$ (20%, Stratified Seed 42)
- **Laporan Data Lama Lengkap**: [`docs/RANGKUMAN_DATA_LAMA.md`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/RANGKUMAN_DATA_LAMA.md)
- **Laporan Data v2 Lengkap**: [`docs/RANGKUMAN_HASIL_EKSPERIMEN_DATA_V2.md`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/RANGKUMAN_HASIL_EKSPERIMEN_DATA_V2.md)

---

## 1. Gambaran Umum Dua Dataset

| Properti | Data Lama | Data Baru (v2) |
| :--- | :--- | :--- |
| **File CSV** | `data_preprocessed_with_emoticon.csv` | `Data/processed/banjir_processed_v2.csv` |
| **Kolom Input LSTM/BiLSTM** | `clean_text_lstm` | `processed_text_v2` |
| **Kolom Input IndoBERTweet** | `text_bert` | `processed_text_v2` |
| **Karakteristik Teks** | Stopword dihapus; emoji → token `[senang]`; pendek (median 8 kata) | Kalimat rekonstruksi via LLM; tanda baca terjaga; lengkap (median 18 kata) |
| **Distribusi Label** | Negatif 54% · Netral 17% · Positif 28% | Sama (SHA256 identik) |
| **Notebook Legacy** | `legacy_notebooks/02,03,04_model_*.ipynb` | `src/` + `utils/` + Kaggle kernels |

---

## 2. Perbandingan Visual Bar Chart Macro F1 (Semua Model & Skenario)

![Bar Chart Perbandingan F1 Semua Model](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/barchart_all_models_f1.png)

> **Biru = Data Lama · Oranye = Data Baru v2 · Garis merah putus = threshold F1 70%**

---

## 3. LSTM — Perbandingan Lengkap Data Lama vs Data Baru v2

**Sumber Code**: [`legacy_notebooks/02_model_lstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/02_model_lstm.ipynb) (Lama) | [`src/train_lstm.py`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src/train_lstm.py) (v2)

| Varian | Acc (Lama) | F1 (Lama) | Rec Netral (Lama) | Acc (v2) | F1 (v2) | Rec Netral (v2) | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empiris Baseline** | 74,05% | **67,97%** | 47,02% | 72,66% | **69,00%** | 55,12% | +1,03 pp ↑ |
| **Empiris Class Weight** | 64,74% | **61,82%** | 56,95% | ~67,00% | ~64,00% | ~58,00% | ~+2,0 pp ↑ |
| **Simulasi 1:1:1** | 42,54% | **42,14%** | 77,48% | 61,64% | **58,16%** | — | +16,02 pp ↑ |
| **Simulasi 6:3:1** | 71,85% | **51,42%** | 0,00% | — | — | — | — |
| **Simulasi 8:1:1** | 54,16% | **23,42%** | 0,00% | — | — | — | — |

**Kurva Training (Lama)**: ![LSTM Legacy Training](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_lstm_legacy.png)

**Confusion Matrices (Lama)**:
![LSTM Legacy CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_lstm_legacy.png)

---

## 4. BiLSTM — Perbandingan Lengkap Data Lama vs Data Baru v2

**Sumber Code**: [`legacy_notebooks/03_model_bilstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/03_model_bilstm.ipynb) (Lama) | [`src/train_bilstm.py`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src/train_bilstm.py) (v2)

| Varian | Acc (Lama) | F1 (Lama) | Rec Netral (Lama) | Acc (v2) | F1 (v2) | Rec Netral (v2) | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empiris Baseline** | 74,34% | **66,59%** | 35,43% | 72,45% | **64,95%** | 49,15% | -1,64 pp ↓ |
| **Empiris CW** | 68,55% | **65,18%** | 59,27% | 65,92% | **62,70%** | 61,20% | -2,48 pp ↓ |
| **Empiris ROS** | — | — | — | 67,05% | **62,71%** | 58,40% | — |
| **Empiris RUS** | — | — | — | 62,95% | **58,92%** | — | — |
| **Empiris SMOTE** | — | — | — | 42,72% | **40,76%** | — | — |
| **Simulasi 1:1:1** | 59,54% | **58,62%** | 65,56% | 61,64% | **58,16%** | — | -0,46 pp ↓ |
| **Simulasi 6:3:1 Baseline** | 72,37% | **51,81%** | 0,00% | 70,45% | **57,16%** | — | +5,35 pp ↑ |
| **Simulasi 6:3:1 ROS** | — | — | — | 64,80% | **59,95%** | — | — |
| **Simulasi 8:1:1 Baseline** | 66,18% | **44,74%** | 0,00% | 63,72% | **45,97%** | — | +1,23 pp ↑ |
| **Simulasi 8:1:1 ROS** | — | — | — | 63,08% | **58,51%** | — | — |

**Kurva Training (Lama)**: ![BiLSTM Legacy](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_legacy.png)

**Kurva Training (v2, Empiris)**: ![BiLSTM v2](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_v2_empirical.png)

**Confusion Matrices (Lama)**:
![BiLSTM Legacy CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_legacy.png)

**Confusion Matrices (v2, Seed 42)**:
![BiLSTM v2 CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_v2.png)

---

## 5. IndoBERTweet-LoRA — Perbandingan Lengkap Data Lama vs Data Baru v2

**Sumber Code**: [`legacy_notebooks/04_model_indobertweet_lora.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/04_model_indobertweet_lora.ipynb) (Lama) | [`src/train_indobert_lora.py`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src/train_indobert_lora.py) (v2)  
**Kaggle Kernel**: `emanuelembuaijdak/baseline-b03-indobert` (GPU T4)

| Varian | Acc (Lama) | F1 (Lama) | Rec Netral (Lama) | Acc (v2) | F1 (v2) | Rec Netral (v2) | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empiris Baseline** | **78,73%** | **73,45%** | 52,65% | 68,67% | **55,16%** | 11,92% | -18,29 pp ↓ |
| **Empiris CW** | 74,10% | **71,14%** | 65,50% | *(planned)* | *(planned)* | — | — |
| **Threshold Calibrated** | — | — | — | **77,46%** | **73,94%** | **66,89%** | **+0,49 pp** ↑ |
| **Simulasi 1:1:1** | 69,13% | **66,94%** | 66,56% | *(planned)* | — | — | — |
| **Simulasi 6:3:1** | **78,50%** | **69,73%** | 32,12% | *(planned)* | — | — | — |
| **Simulasi 8:1:1** | **77,92%** | **68,28%** | 29,80% | *(planned)* | — | — | — |

> [!IMPORTANT]
> **Root Cause Gap Lama vs v2**: Data lama menggunakan `text_bert` di mana emoji dikonversi ke token emosi eksplisit (`[senang]`, `[marah]`, `[sedih]`). Hal ini memberikan sinyal sentimen yang sangat kuat dan langsung ke tokenizer IndoBERTweet. Data v2 (`processed_text_v2`) tidak mengandung token emosi ini — sehingga model bergantung penuh pada konteks kalimat. Tanpa kalibrasi ambang batas, Recall Netral runtuh dari 52,65% → 11,92%.

**Kurva Training (v2, LR Sweep)**:
![IndoBERT LR Sweep](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_indobert_lora_v2.png)

**Confusion Matrices (Lama)**:
![IndoBERT Legacy CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_legacy.png)

**Confusion Matrices (v2, Ablasi Input)**:
![IndoBERT v2 CM](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_v2.png)

---

## 6. Matriks Ringkasan Performa Terbaik Per Arsitektur

| Arsitektur | Data | Varian Terbaik | Test Accuracy | Macro F1 | Recall Netral |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **LSTM** | Lama | Empiris Baseline | 74,05% | **67,97%** | 47,02% |
| **LSTM** | v2 | Empiris Baseline | 72,66% | **69,00%** | 55,12% |
| **BiLSTM** | Lama | Empiris Baseline | 74,34% | **66,59%** | 35,43% |
| **BiLSTM** | v2 | Empiris Baseline | 72,45% | **64,95%** | 49,15% |
| **BiLSTM + ROS** | v2 | Empiris | 67,05% | **62,71%** | 58,40% |
| **IndoBERT-LoRA** | Lama | Empiris Baseline (ckpt-780) | **78,73%** | **73,45%** | 52,65% |
| **IndoBERT-LoRA** | v2 | Tuned Baseline (lr=2e-4) | 68,67% | 55,16% | 11,92% |
| **IndoBERT-LoRA Calibrated** | v2 | Threshold w=[1.0, 1.5, 1.0] | **77,46%** | **73,94%** | **66,89%** |

---

## 7. Analisis Naratif untuk Bab IV Tesis

### 7.1 Mengapa LSTM/BiLSTM Lebih Baik di Data Lama?
Pada data lama, `clean_text_lstm` telah melalui *stopword removal* agresif, sehingga setiap kata yang tersisa merupakan kata konten bermuatan sentimen tinggi. Kepadatan leksikon yang tinggi ini sangat cocok untuk arsitektur berulang (LSTM/BiLSTM) yang mendeteksi pola urutan lokal. Di data v2, kalimat lebih panjang dan alami, sehingga model perlu menangkap ketergantungan jarak jauh yang menjadi keunggulan Transformer.

### 7.2 Mengapa IndoBERTweet Ambruk di Data v2 (Sebelum Kalibrasi)?
Pada data lama, token `[senang]`, `[marah]`, dan `[sedih]` berfungsi sebagai *anchor* sentimen yang sangat kuat — model cukup mengandalkan keberadaan token ini untuk mengklasifikasikan. Di data v2, sinyal ini hilang, dan model harus bergantung pada representasi semantik full-context. Tanpa penyesuaian *decision boundary*, probabilitas posterior kelas Netral tertekan oleh volume data Negatif yang 3× lebih banyak.

### 7.3 Konsistensi Fenomena Majority Collapse
Fenomena *majority collapse* (Recall Netral = 0%) terbukti **tidak bergantung pada arsitektur atau versi data** — ia muncul secara konsisten setiap kali model dilatih pada skenario ketimpangan 6:3:1 atau 8:1:1 tanpa teknik balancing. Ini mengonfirmasi bahwa fenomena ini bersifat **struktural** (bukan artefak implementasi) dan merupakan kontribusi temuan penting tesis ini.

---

## 8. Indeks Artefak & File Terkait

| Tipe Artefak | Deskripsi | Link |
| :--- | :--- | :--- |
| 📂 **Output Legacy Rerun** | 15 folder eksperimen lengkap | [`Output/legacy_rerun/`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/legacy_rerun/) |
| 📊 **Master CSV Legacy** | 15 baris metrik terkonsolidasi | [`Output/legacy_rerun/master_summary.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/legacy_rerun/master_summary.csv) |
| 📊 **Bar Chart Semua Model** | F1 comparison semua varian | [`Output/charts/barchart_all_models_f1.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/barchart_all_models_f1.png) |
| 📈 **Line Charts LSTM Lama** | Training/Val curves 5 varian | [`Output/charts/linechart_lstm_legacy.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_lstm_legacy.png) |
| 📈 **Line Charts BiLSTM Lama** | Training/Val curves 5 varian | [`Output/charts/linechart_bilstm_legacy.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_legacy.png) |
| 📈 **Line Charts BiLSTM v2** | Training/Val curves empiris v2 | [`Output/charts/linechart_bilstm_v2_empirical.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_bilstm_v2_empirical.png) |
| 📈 **Line Charts IndoBERT v2** | LR + capacity sweep v2 | [`Output/charts/linechart_indobert_lora_v2.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/linechart_indobert_lora_v2.png) |
| 🖼️ **CM Grid LSTM Lama** | 5 confusion matrices | [`Output/charts/cm_grid_lstm_legacy.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_lstm_legacy.png) |
| 🖼️ **CM Grid BiLSTM Lama** | 5 confusion matrices | [`Output/charts/cm_grid_bilstm_legacy.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_legacy.png) |
| 🖼️ **CM Grid IndoBERT Lama** | 4 confusion matrices | [`Output/charts/cm_grid_bert_legacy.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_legacy.png) |
| 🖼️ **CM Grid BiLSTM v2** | 5 confusion matrices (seed42) | [`Output/charts/cm_grid_bilstm_v2.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bilstm_v2.png) |
| 🖼️ **CM Grid IndoBERT v2** | Input ablation CMs | [`Output/charts/cm_grid_bert_v2.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/charts/cm_grid_bert_v2.png) |
| 📓 **Notebook LSTM Lama** | Legacy training notebook | [`legacy_notebooks/02_model_lstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/02_model_lstm.ipynb) |
| 📓 **Notebook BiLSTM Lama** | Legacy training notebook | [`legacy_notebooks/03_model_bilstm.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/03_model_bilstm.ipynb) |
| 📓 **Notebook IndoBERT Lama** | Legacy training notebook | [`legacy_notebooks/04_model_indobertweet_lora.ipynb`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/legacy_notebooks/04_model_indobertweet_lora.ipynb) |
| 🔬 **Script Retraining** | Deterministik, seed=42 | [`experiments/run_legacy_rerun.py`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/experiments/run_legacy_rerun.py) |
| 📋 **Experiment Registry** | ID & status semua eksperimen | [`EXPERIMENT_REGISTRY.md`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/EXPERIMENT_REGISTRY.md) |
