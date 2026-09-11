# 📊 Ringkasan Master Evaluasi Model Tesis (Single Source of Truth)

Dokumen ini dihasilkan secara dinamis dan otomatis dari hasil eksekusi nyata (*live execution*) seluruh 7 varian model:
- **Model 1 s.d. 5**: Dieksekusi secara live di workstation lokal (LSTM Baseline, Class Weight, ROS, RUS, SMOTE).
- **Model 6 & 7**: Dieksekusi pada Kaggle Cloud GPU (IndoBERTweet-LoRA Vanilla & TAPT 2-Stage Transfer Learning).

---

## 1. Master Comparison Table (Data Latih vs Data Uji Terkunci $n=1.730$)

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Train F1 (%) | Macro F1 (%) | Generalization Gap Acc (%) | Recall Netral (%) |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance Data) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | 73.19% | **61.56%** | +10.40% | 28.81% |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive Learning) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | 85.06% | **64.60%** | +16.81% | 40.40% |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | 92.44% | **64.89%** | +22.40% | 48.68% |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | 70.64% | **52.72%** | +17.53% | 56.62% |
| 5 | **LSTM SMOTE** | SMOTE (Synthetic Minority Over-sampling Technique) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | 64.33% | **57.37%** | +0.63% | 43.71% |
| 6 | **IndoBERTweet-LoRA** | Natural Baseline | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | 80.20% | **73.90%** | +6.17% | 61.26% |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | 82.50% | **74.61%** | +6.34% | 52.65% |

---

## 2. Temuan Ilmiah Utama & Pembahasan untuk Bab IV Tesis

### A. Mengapa Imbalance Data Alami Mencapai Akurasi Global Lebih Tinggi dari RUS dan SMOTE?
1. **The Accuracy Paradox**:
   - Baseline alami bias ke kelas mayoritas (Positif/Negatif mencakup 82% data uji), sehingga mudah mencatat akurasi 70,92%. Namun, model menderita **Majority Collapse** dengan Recall Netral hanya **28,81%**.
2. **Information Loss pada RUS**:
   - Random Undersampling membuang lebih dari 45% data mayoritas, mengakibatkan model kehilangan wawasan kosakata dan akurasi jatuh ke **53,06%**.
3. **Discrete Token Corruption pada SMOTE**:
   - Interpolasi numerik SMOTE pada sekuens integer token merusak representasi kalimat, menurunkan akurasi ke **64,39%**.
4. **Keberhasilan Penyeimbangan pada Kelas Minoritas**:
   - Penyeimbangan data **berhasil menyelamatkan kelas minoritas (Netral)**: Recall Netral melonjak dari **28,81% (Baseline)** ke **40,40% (Class Weight)** dan **48,68% (ROS)**.

### B. Superioritas Mutlak IndoBERTweet-LoRA & TAPT
- **IndoBERTweet-LoRA Vanilla** menembus akurasi **77,98%** dan Macro F1 **73,90%**, dengan Recall Netral mencapai **61,26%**.
- **TAPT IndoBERTweet-LoRA** menjadi model terbaik mutlak (**Akurasi 80,06%** dan **Macro F1 74,61%**), membuktikan bahwa adaptasi leksikon bencana (MLM 3 epoch) memberikan representasi semantik superior tanpa perlu manipulasi frekuensi sampel buatan.
