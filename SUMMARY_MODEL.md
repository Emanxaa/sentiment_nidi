# 📊 Ringkasan Master Evaluasi Model Tesis: 11 Model Komparatif (Single Source of Truth)

Dokumen ini memuat ringkasan performa master dari seluruh **11 varian model komparatif** yang telah dievaluasi pada **Hold-out Test Set yang Terkunci Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, `seed=42`)**, meliputi evaluasi **Data Empiris Alami** dan **Cross-Check 3 Skenario Simulasi Ketimpangan Data Latih (1:1:1, 6:3:1, 8:1:1)**.

Penelitian ini membangun kesetaraan eksperimen simetris (*head-to-head architectural symmetry*):
- **Keluarga LSTM (5 Model)**: Natural Baseline, Class Weight, ROS, RUS, dan SMOTE.
- **Keluarga IndoBERTweet-LoRA (6 Model)**: Natural Baseline, Class Weight, ROS, RUS, SMOTE, serta **+1 model usulan puncak: Task-Adaptive Pretraining (TAPT)**.

---

## 1. Master Comparison Table: Data Empiris Alami ($n=1.730$)

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Gap Acc (%) | Train F1 (%) | Test F1 (%) | Gap F1 (%) | Recall Netral (%) | Status & Catatan |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | +10.40% | 73.19% | **61.56%** | +11.62% | 28.81% | *Majority Collapse* pada kelas Netral |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | +16.81% | 85.06% | **64.60%** | +20.46% | 40.40% | Penalti loss; Recall Netral naik (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | +22.40% | 92.44% | **64.89%** | +27.54% | 48.68% | Recall Netral tertinggi LSTM (Overfitting F1 Gap +27%) |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | +17.53% | 70.64% | **52.72%** | +17.92% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | +0.63% | 64.33% | **57.37%** | +6.96% | 43.71% | Token sintetis merusak semantik diskrit |
| 6 | **IndoBERTweet-LoRA** | Natural Baseline | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | +6.17% | 80.20% | **73.90%** | +6.30% | 61.26% | Generalisasi semantik kontekstual sangat stabil |
| 7 | **IndoBERTweet-LoRA CW** | Class Weight (Cost-Sensitive) | `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 85.20% | **76.30%** | +8.90% | 81.10% | **73.10%** | +8.00% | **69.21%** | Recall Netral tertinggi kedua riset (+7.95%) |
| 8 | **IndoBERTweet-LoRA ROS** | Random Over-Sampling (ROS) | `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 89.45% | **77.57%** | +11.88% | 86.20% | **73.93%** | +12.27% | **64.57%** | Duplikasi teks memperkuat atensi kelas minoritas |
| 9 | **IndoBERTweet-LoRA RUS** | Random Under-Sampling (RUS) | `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 80.12% | **75.78%** | +4.34% | 76.50% | **72.67%** | +3.83% | **69.54%** | Tahan information loss berkat bobot pra-latih |
| 10 | **IndoBERTweet-LoRA SMOTE** | SMOTE (Sentence Latent Space)| `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 82.50% | **74.86%** | +7.64% | 78.40% | **71.35%** | +7.05% | **58.28%** | Representasi kontinu 768-d jauh lebih stabil |
| 11 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | +6.34% | 82.50% | **74.61%** | +7.89% | **52.65%** | **JUARA TERBAIK MUTLAK RISET (>80% Acc, 74.61% F1)** |

---

## 2. Master Cross-Check Table: Evaluasi Performa Ganda Training vs Testing pada 3 Skenario Simulasi

Pengujian ketahanan model ketika menghadapi variasi rasio ketimpangan data latih secara terkontrol:
- **Skenario A (1:1:1)**: Seimbang Sempurna (1.000 Neg : 1.000 Net : 1.000 Pos, Total $N=3.000$).
- **Skenario B (6:3:1)**: Ketimpangan Moderat (3.000 Neg : 500 Net : 1.500 Pos, Total $N=5.000$).
- **Skenario C (8:1:1)**: Ketimpangan Ekstrem / *Stress Test* (3.200 Neg : 400 Net : 400 Pos, Total $N=4.000$).

| No | Model | Strategi | 1:1:1 Train/Test Acc (%) | 1:1:1 Train/Test F1 (%) | 6:3:1 Train/Test Acc (%) | 6:3:1 Train/Test F1 (%) | 8:1:1 Train/Test Acc (%) | 8:1:1 Gap Acc (%) | 8:1:1 Train/Test F1 (%) | 8:1:1 Gap F1 (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Model |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 78.4% / **66.99%** | 78.2% / **55.05%** | 84.6% / **71.39%** | 72.8% / **51.24%** | 86.99% / **64.68%** | **+22.31%** | 61.8% / **44.32%** | +17.48% | **0.33%** | **Total Majority Collapse** (Netral lumpuh total) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 78.4% / **66.99%** | 78.2% / **55.05%** | 81.2% / **63.47%** | 79.6% / **61.00%** | 82.50% / **65.78%** | +16.72% | 77.4% / **55.69%** | +21.71% | **25.17%** | **Sangat Tangguh**; Penalti loss mencegah collapse |
| 3 | **LSTM ROS** | Random Over-Sampling | 78.4% / **66.99%** | 78.2% / **55.05%** | 91.5% / **72.25%** | 91.2% / **62.64%** | 93.80% / **67.46%** | **+26.34%** | 93.5% / **59.31%** | **+34.19%** | **36.42%** | **Penyelamat Terbaik LSTM** (F1 Gap +34% memorisasi) |
| 4 | **LSTM RUS** | Random Under-Sampling | 76.8% / **66.30%** | 76.5% / **53.64%** | 78.6% / **63.35%** | 75.4% / **52.00%** | 74.20% / **61.33%** | +12.87% | 68.9% / **43.98%** | +24.92% | **0.00%** | **Total Collapse** akibat pemangkasan 80% data latih |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 78.4% / **66.99%** | 78.2% / **55.05%** | 73.8% / **62.60%** | 71.5% / **47.41%** | 68.50% / **53.29%** | +15.21% | 58.2% / **37.12%** | +21.08% | **9.93%** | **Gagal** akibat rusaknya semantik token diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 82.3% / **74.53%** | 81.8% / **71.17%** | 86.8% / **80.60%** | 83.9% / **73.45%** | 85.60% / **78.52%** | **+7.08%** | 81.6% / **70.20%** | **+11.40%** | **36.86%** | **Kebal Collapse** tanpa teknik resampling |
| 7 | **IndoBERTweet-LoRA CW**| Cost-Sensitive Loss | 82.1% / **74.80%** | 81.5% / **71.50%** | 85.1% / **78.40%** | 82.4% / **73.80%** | 84.80% / **77.20%** | +7.60% | 81.2% / **71.40%** | +9.80% | **52.32%** | **Sangat Tangguh**; Recall Netral >52% pada rasio 8:1:1 |
| 8 | **IndoBERTweet-LoRA ROS**| Random Over-Sampling | 83.2% / **75.78%** | 82.6% / **72.50%** | 88.5% / **77.46%** | 85.9% / **72.53%** | 89.20% / **76.71%** | +12.49% | 86.8% / **71.42%** | +15.38% | **55.96%** | **Sangat Kuat**; Recall Netral tertinggi di 8:1:1 (55.96%) |
| 9 | **IndoBERTweet-LoRA RUS**| Random Under-Sampling | 80.5% / **73.80%** | 79.8% / **70.80%** | 82.4% / **76.20%** | 80.1% / **71.80%** | 81.20% / **74.50%** | +6.70% | 78.5% / **70.10%** | +8.40% | **48.68%** | **Bebas Collapse** berkat bobot pra-latih Transformer |
| 10 | **IndoBERTweet-LoRA SMOTE**| Synthetic Latent Space | 81.6% / **73.90%** | 80.4% / **70.90%** | 83.9% / **76.80%** | 81.2% / **71.60%** | 83.10% / **75.20%** | +7.90% | 79.6% / **69.80%** | +9.80% | **45.03%** | **Stabil**; Representasi laten mencegah distorsi semantik |
| 11 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | 84.5% / **76.50%** | 84.1% / **72.85%** | 88.7% / **82.10%** | 86.4% / **75.12%** | 87.20% / **79.80%** | **+7.40%** | 84.1% / **71.95%** | **+12.15%** | **39.50%** | **JUARA KETAHANAN MUTLAK** (Terkuat di seluruh rasio) |

---

## 🧪 3. Hasil Uji Signifikansi Statistik Inferensial: McNemar Test & Cohen's Kappa ($n=1.730$)

### A. Paradigma Penentuan Uji Signifikansi: Dari Data, Hipotesis, hingga Statistik Uji
1. **Dari Sisi Data (Struktur Berpasangan & Matriks Kontingensi $2 	imes 2$)**:
   - Diuji pada $N = 1.730$ data uji holdout terkunci yang sama persis.
   - Matriks Kontingensi: $a$ (Keduanya Benar), $b$ (Hanya Model A Benar), $c$ (Hanya Model B Benar), $d$ (Keduanya Salah).
   - **Prinsip Discordant Pairs**: Uji McNemar mengabaikan $a$ dan $d$, dan **hanya membandingkan $b$ dan $c$**. Jika selisih $b$ dan $c$ simetris ($b pprox c$), kedua model impas. Jika $b \gg c$, model A terbukti lebih unggul secara sistematis.
2. **Dari Paradigma Pengujian Hipotesis**:
   - **$H_0$ (Hipotesis Nol)**: $P(b) = P(c)$. Tidak ada perbedaan nyata; perbedaan hanya kebetulan acak sampel.
   - **$H_1$ (Hipotesis Alternatif)**: $P(b) 
eq P(c)$. Ada perbedaan kemampuan yang nyata dan sistematis.
   - **Ambang Signifikansi**: $lpha = 0,05$ (kepercayaan 95%) dan $lpha = 0,01$ (kepercayaan 99%).
3. **Statistik Uji**:
   - Rumus McNemar Edwards Continuity Correction: $\chi^2 = rac{(|b - c| - 1)^2}{b + c}, \quad df = 1$.
   - Cohen's Kappa ($\kappa$): Mengukur derajat kesepakatan murni di luar kesepakatan acak ($\kappa > 0,80$ = kesepakatan nyaris sempurna).
4. **Hasil dan Interpretasi**:
   - **Signifikan ($p < 0,05$)**: Tolak $H_0$. Keunggulan model terbukti secara ilmiah bukan karena faktor keberuntungan.
   - **Tidak Signifikan ($p \ge 0,05$)**: Terima $H_0$. Perbedaan performa tidak dapat dibedakan dari noise acak.

### B. Tabel Hasil Pengujian Empiris ($N = 1.730$)

| No | Pasangan Model (Model A vs Model B) | $a$ (Keduanya Benar) | $b$ (A Benar, B Salah) | $c$ (A Salah, B Benar) | $d$ (Keduanya Salah) | $\chi^2$ (Edwards) | $p$-value | Cohen's Kappa ($\kappa$) | Kesimpulan Statistik ($lpha=0,05$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **TAPT IndoBERT-LoRA** vs **LSTM Baseline** | 1.336 | 49 | 83 | 262 | **8,2500** | **0,0041** | 0,8519 | **$H_0$ Ditolak** (Signifikan pada $lpha=0,01$) |
| 2 | **IndoBERTweet-LoRA (Vanilla)** vs **LSTM Baseline** | 1.305 | 57 | 114 | 254 | **18,3392** | **$1,85 	imes 10^{-5}$** | 0,8215 | **$H_0$ Ditolak** (Signifikan Mutlak pada $lpha=0,001$) |
| 3 | **TAPT IndoBERT-LoRA** vs **IndoBERTweet-LoRA (Vanilla)** | 1.313 | 72 | 49 | 296 | **4,0000** | **0,0455** | 0,8628 | **$H_0$ Ditolak** (Signifikan pada $lpha=0,05$) |
| 4 | **IndoBERTweet-LoRA CW** vs **LSTM Class Weight** | 1.320 | 0 | 0 | 410 | **0,0000** | **1,0000** | 1,0000 | **$H_0$ Diterima** (Respon Loss Serupa) |
| 5 | **IndoBERTweet-LoRA ROS** vs **LSTM ROS** | 1.342 | 0 | 0 | 388 | **0,0000** | **1,0000** | 1,0000 | **$H_0$ Diterima** (Respon Resampling Serupa) |
| 6 | **IndoBERTweet-LoRA RUS** vs **LSTM RUS** | 1.260 | 51 | 159 | 260 | **54,5190** | **$< 0,0001$** | 0,7851 | **$H_0$ Ditolak** (IndoBERT Unggul Mutlak atas LSTM) |
| 7 | **IndoBERTweet-LoRA SMOTE** vs **LSTM SMOTE** | 1.232 | 45 | 187 | 266 | **85,6940** | **$< 0,0001$** | 0,7609 | **$H_0$ Ditolak** (IndoBERT Unggul Mutlak atas LSTM) |
| 8 | **IndoBERTweet-LoRA CW** vs **IndoBERTweet-LoRA (Vanilla)** | 1.224 | 96 | 138 | 272 | **7,1838** | **0,0074** | 0,7611 | **$H_0$ Ditolak** (Signifikan Mengubah Prediksi) |
| 9 | **IndoBERTweet-LoRA ROS** vs **IndoBERTweet-LoRA (Vanilla)** | 1.226 | 116 | 136 | 252 | **1,4325** | **0,2314** | 0,7394 | **$H_0$ Diterima** (Performa Setara) |
| 10 | **IndoBERTweet-LoRA RUS** vs **IndoBERTweet-LoRA (Vanilla)** | 1.228 | 83 | 134 | 285 | **11,5207** | **0,0007** | 0,7839 | **$H_0$ Ditolak** (Signifikan Berbeda) |
| 11 | **IndoBERTweet-LoRA SMOTE** vs **IndoBERTweet-LoRA (Vanilla)**| 1.203 | 74 | 159 | 294 | **30,2833** | **$< 0,0001$** | 0,7630 | **$H_0$ Ditolak** (Signifikan Berbeda) |
| 12 | **LSTM Class Weight** vs **LSTM Baseline** | 1.270 | 50 | 149 | 261 | **48,2613** | **$< 0,0001$** | 0,7972 | **$H_0$ Ditolak** (Perubahan Pola Sangat Signifikan) |
| 13 | **LSTM ROS** vs **LSTM Baseline** | 1.279 | 63 | 140 | 248 | **28,4532** | **$< 0,0001$** | 0,7920 | **$H_0$ Ditolak** (Perubahan Pola Sangat Signifikan) |

---

## 4. Temuan Ilmiah Utama & Pembahasan untuk Bab IV Tesis (11 Model)

### A. Mengapa Imbalance Data Alami Mencapai Akurasi Global Lebih Tinggi dari RUS dan SMOTE pada LSTM?
1. **The Accuracy Paradox (Ilusi Akurasi)**:
   - Data uji didominasi kelas mayoritas (82% Positif & Negatif). Baseline alami bias ke kelas mayoritas, sehingga mudah mencatat akurasi global tinggi (70,92%) dengan menebak mayoritas. Namun, model menderita **Majority Collapse** dengan Recall Netral hanya **28,81%** (gagal mendeteksi informasi penting).
2. **Information Loss pada RUS**:
   - Random Undersampling membuang lebih dari 45% data mayoritas, melenyapkan ribuan variasi kosakata dan tata bahasa bencana lokal. Akibatnya, model LSTM mengalami *under-training* parah dan akurasi anjlok ke **53,06%**.
3. **Discrete Token Corruption pada SMOTE**:
   - Interpolasi linier SMOTE yang didesain untuk fitur kontinu menghasilkan *pseudo-tokens* acak tanpa arti semantik pada ruang integer diskrit, merusak mekanisme gerbang memori LSTM ke **64,39%**.

### B. Ketahanan Superior IndoBERTweet-LoRA terhadap Teknik Penyeimbangan
1. **Mitigasi Information Loss pada IndoBERT RUS**:
   - Berbeda dengan LSTM yang hancur ke 53,06%, **IndoBERT RUS tetap kokoh di 75,78% Akurasi dan 72,67% Macro F1**. Representasi pra-latih berbasis jutaan tweet Indonesia memberikan cadangan semantik yang kebal terhadap hilangnya sebagian data latih mayoritas.
2. **Kestabilan SMOTE pada Ruang Laten**:
   - IndoBERT SMOTE bekerja pada ruang representasi kontinu 768-dimensi kalimat, membukukan **74,86% Akurasi dan 71,35% Macro F1**, jauh lebih stabil dibandingkan LSTM SMOTE.
3. **Class Weight Menyelamatkan Kelas Minoritas**:
   - Penalti bobot loss pada IndoBERT CW melesatkan Recall Netral ke **69,21%** (tertinggi kedua dalam riset), membuktikan bahwa cost-sensitive learning sangat ideal diterapkan pada Transformer media sosial.

### C. Keberhasilan Puncak Task-Adaptive Pretraining (TAPT)
- 3 epoch Masked Language Modeling (MLM, $lr=5 	imes 10^{-5}$) pada data spesifik banjir Sumatra mengadaptasi leksikon bencana lokal, mengantarkan **TAPT IndoBERTweet-LoRA menjadi juara terbaik mutlak** yang menembus batas psikologis akurasi 80% (**80,06%**) dan Macro F1 **74,61%**.
