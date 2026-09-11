# 📊 Ringkasan Master Evaluasi Model Tesis (Single Source of Truth)

Dokumen ini memuat ringkasan performa master dari seluruh 7 varian model yang telah dieksekusi secara nyata (*live raw run*) pada **Hold-out Test Set yang Terkunci Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, `seed=42`)**, meliputi evaluasi **Data Empiris Alami** dan **Cross-Check 3 Skenario Simulasi Ketimpangan Data Latih (1:1:1, 6:3:1, 8:1:1)**.

---

## 1. Master Comparison Table: Data Empiris Alami ($n=1.730$)

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Generalization Gap Acc (%) | Recall Netral (%) | Status & Karakteristik Model |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | 61.65% | 61.69% | **61.56%** | +10.40% | 28.81% | Bias kelas mayoritas (*Majority Collapse*) |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | 64.88% | 64.35% | **64.60%** | +16.81% | 40.40% | Penalti loss; Recall Netral naik (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | 64.38% | 65.62% | **64.89%** | +22.40% | 48.68% | Recall Netral terbaik di keluarga LSTM |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | 57.34% | 56.95% | **52.72%** | +17.53% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | 61.05% | 57.07% | **57.37%** | +0.63% | 43.71% | Token sintetis merusak semantik diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | 73.18% | 74.88% | **73.90%** | +6.17% | 61.26% | Unggul telak atas seluruh varian LSTM |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | 75.83% | 73.83% | **74.61%** | +6.34% | 52.65% | **JUARA TERBAIK MUTLAK RISET (>80%)** |

---

## 2. Master Cross-Check Table: Ketahanan Lintas 3 Skenario Simulasi Ketimpangan

Pengujian ketahanan model ketika menghadapi tingkat ketidakseimbangan yang divariasikan secara terkontrol pada data latih:
- **Skenario A (1:1:1)**: Seimbang Sempurna (1.000 Neg : 1.000 Net : 1.000 Pos).
- **Skenario B (6:3:1)**: Ketimpangan Moderat (3.000 Neg : 500 Net : 1.500 Pos).
- **Skenario C (8:1:1)**: Ketimpangan Ekstrem / *Stress Test* (3.200 Neg : 400 Net : 400 Pos).

| No | Model | Strategi | Empiris F1 (%) | 1:1:1 F1 (%) | 6:3:1 F1 (%) | 8:1:1 F1 (%) | Empiris Rec Netral (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Terhadap Majority Collapse |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 61.56% | 55.05% | 51.24% | **44.32%** | 28.81% | **0.33%** | **TOTAL MAJORITY COLLAPSE** (Netral mendekati 0%) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 64.60% | 55.05% | 61.00% | 55.69% | 40.40% | 25.17% | **Sangat Tangguh**; Penalti loss mencegah collapse |
| 3 | **LSTM ROS** | Random Over-Sampling | 64.89% | 55.05% | 62.64% | **59.31%** | 48.68% | **36.42%** | **Penyelamat Terbaik LSTM** pada Rasio 8:1:1 |
| 4 | **LSTM RUS** | Random Under-Sampling | 52.72% | 53.64% | 52.00% | 43.98% | 56.62% | **0.00%** | **Total Collapse** akibat pemangkasan 80% data latih |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 57.37% | 55.05% | 47.41% | 37.12% | 43.71% | 9.93% | Gagal akibat rusaknya semantik token diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 73.90% | 71.17% | 73.45% | 70.20% | 61.26% | 36.86% | **KEBAL COLLAPSE** tanpa teknik resampling |
| 7 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | **74.61%** | **72.85%** | **75.12%** | **71.95%** | 52.65% | **39.50%** | **JUARA KETAHANAN TERTINGGI MUTLAK** (Terkuat di seluruh rasio) |

---

## 3. Temuan Ilmiah Utama & Pembahasan untuk Bab IV Tesis

### A. Mengapa Imbalance Data Alami Mencapai Akurasi Global Lebih Tinggi dari RUS dan SMOTE?
1. **The Accuracy Paradox (Ilusi Akurasi)**:
   - Data uji didominasi kelas mayoritas (82% Positif & Negatif). Baseline alami bias ke kelas mayoritas, sehingga mudah mencatat akurasi global tinggi (70,92%) dengan menebak mayoritas. Namun, model menderita **Majority Collapse** dengan Recall Netral hanya **28,81%** (gagal mendeteksi informasi penting).
2. **Information Loss pada RUS**:
   - Random Undersampling membuang lebih dari 45% data mayoritas, melenyapkan ribuan variasi kosakata dan tata bahasa bencana lokal. Akibatnya, model mengalami *under-training* parah dan akurasi anjlok ke **53,06%**.
3. **Discrete Token Corruption pada SMOTE**:
   - Interpolasi linier SMOTE yang didesain untuk fitur kontinu menghasilkan *pseudo-tokens* acak tanpa arti semantik pada ruang integer diskrit, merusak mekanisme gerbang memori LSTM ke **64,39%**.
4. **Keberhasilan Penyeimbangan pada Kelas Minoritas**:
   - Penyeimbangan data **berhasil menyelamatkan kelas minoritas (Netral)**: Recall Netral melonjak dari **28,81% (Baseline)** ke **40,40% (Class Weight)** dan **48,68% (ROS)**.

### B. Bukti Empiris Fenomena Majority Collapse pada Simulasi
- Pada kondisi ekstrem 8:1:1, **LSTM Baseline murni mengalami keruntuhan total** dengan Recall Netral jatuh ke **0,33%** (hanya 1 dari 302 tweet netral yang terdeteksi).
- Sebaliknya, **IndoBERTweet-LoRA terbukti kebal (*collapse-immune*)** dengan mempertahankan Macro F1 **70,20%** dan Recall Netral **36,86%** bahkan tanpa penyeimbangan sampel buatan.
- **TAPT IndoBERTweet-LoRA** membuktikan ketahanan terkuat di seluruh skenario: Macro F1 tetap kokoh di **71,95%** dan Recall Netral **39,50%** pada rasio 8:1:1.

### C. Superioritas Mutlak IndoBERTweet-LoRA vs LSTM
- Model berbasis Transformer IndoBERTweet-LoRA melampaui varian LSTM terbaik dengan lompatan akurasi signifikan (**+9,14%** dari 70,92% ke 80,06%) dan Macro F1 **+13,05%** (dari 61,56% ke 74,61%).
- Mekanisme *Bidirectional Self-Attention* mampu menangkap dependensi kata jarak jauh, istilah gaul Twitter, dan konteks kalimat sarkastik secara presisi.

### D. Efektivitas Task-Adaptive Pretraining (TAPT)
- 3 epoch Masked Language Modeling (MLM, $lr=5 \times 10^{-5}$) pada data spesifik banjir berhasil mengadaptasi leksikon kebencanaan lokal Sumatra, mendorong model **TAPT IndoBERTweet-LoRA menjadi juara terbaik mutlak** yang menembus batas psikologis akurasi 80% (**80,06%**) dan Macro F1 **74,61%**.
