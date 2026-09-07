# 09 — Rangkuman Master Model & Sintesis Eksperimen Tesis

Dokumen ini memuat kesimpulan komprehensif dan tabel komparasi dari seluruh varian model yang telah dikembangkan dalam Tesis Analisis Sentimen Bencana Banjir:
1. **Data Processing**: Rekonstruksi teks terpotong via LLM + Regex + Normalisasi Leksikon + WordCloud.
2. **LSTM Imbalance Data** (Natural Baseline).
3. **LSTM Balance Data - Class Weight** (Cost-Sensitive Learning).
4. **LSTM Balance Data - Oversampling** (Random Over-Sampling / ROS).
5. **LSTM Balance Data - Undersampling** (Random Under-Sampling / RUS).
6. **LSTM Balance Data - SMOTE** (Synthetic Minority Over-sampling Technique).
7. **IndoBERTweet-LoRA** (Vanilla Adapter Tuning).
8. **TAPT IndoBERTweet-LoRA** (Task-Adaptive Pretraining + LoRA).

---

## 1. Master Comparison Table Seluruh Model

Evaluasi dilakukan secara adil pada **Data Uji Terkunci yang Sama Persis ($n = 1.730$ tweet, 20% Stratified Split, Seed 42)**:

| No | Nama Model / Eksperimen | Strategi Balancing | Arsitektur / Backbone | Hiperparameter Utama | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Recall Netral (%) |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **LSTM Baseline** | Natural (Imbalance) | Embedding(128) + LSTM(64) | lr: 2e-4, bs: 16, drop: 0.2 | 72.45% | 66.82% | 63.45% | 64.95% | 48.20% |
| 2 | **LSTM Class Weight** | Class Weight (Inverse Freq) | Embedding(128) + LSTM(64) | lr: 2e-4, bs: 16, drop: 0.2 | 71.21% | 63.50% | 64.10% | 63.26% | 55.40% |
| 3 | **LSTM Oversampling** | Random Over-Sampling (ROS) | Embedding(128) + LSTM(64) | lr: 2e-4, bs: 16, drop: 0.2 | 72.83% | 65.40% | 64.30% | 64.81% | 52.10% |
| 4 | **LSTM Undersampling** | Random Under-Sampling (RUS) | Embedding(128) + LSTM(64) | lr: 2e-4, bs: 16, drop: 0.2 | 68.96% | 61.20% | 63.80% | 62.01% | 56.80% |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | Embedding(128) + LSTM(64) | lr: 2e-4, bs: 16, drop: 0.2 | 71.85% | 64.10% | 64.20% | 64.12% | 51.30% |
| 6 | **IndoBERT-LoRA** | Natural Baseline | `indobertweet-base-uncased` | r: 16, a: 32, lr: 2e-4, ep: 5 | 78.73% | 75.12% | 72.30% | 73.45% | 53.58% |
| 7 | **TAPT IndoBERT-LoRA**| Domain Adaptation (MLM) | `indobertweet` + TAPT (3 ep) | MLM lr: 5e-5, FT lr: 2e-4 | **79.48%** | **76.45%** | **74.10%** | **74.92%** | **61.20%** |

---

## 2. Sintesis Temuan Ilmiah & Pembahasan (Bab IV Tesis)

### A. Perbandingan RNN (LSTM) vs Transformer (IndoBERTweet)
- **Superioritas Kontekstual**: IndoBERTweet-LoRA mengungguli seluruh varian LSTM dengan selisih akurasi sebesar **+6% hingga +10%** dan Macro F1 **+8% hingga +11%**.
- **Mekanisme Atensi vs Sekuensial**: LSTM terbatas pada pemrosesan berurutan satu arah (*unidirectional*) yang rentan mengalami degradasi memori (*forgetting*) pada kalimat panjang. Sebaliknya, mekanisme *Self-Attention* pada IndoBERTweet mampu menangkap relasi kata jarak jauh dan nuansa sarkasme khas Twitter.

### B. Analisis 5 Strategi Penyeimbangan Data pada LSTM
1. **Trade-off Akurasi vs Recall Kelas Minoritas**:
   - Varian **Class Weight** dan **Random Undersampling (RUS)** sukses mendongkrak Recall kelas Netral (naik dari 48.20% ke **55.40%** dan **56.80%**).
   - Namun, kenaikan recall minoritas ini diiringi oleh penurunan akurasi global (RUS turun ke 68.96%) karena hilangnya informasi representatif dari kelas mayoritas (*information loss*).
2. **Kestabilan Random Oversampling (ROS)**:
   - ROS memberikan performa paling seimbang di antara keluarga LSTM (Akurasi 72.83%, Macro F1 64.81%), karena mempertahankan seluruh variasi kosakata data latih asli.
3. **Keterbatasan SMOTE pada Data Sekuensial**:
   - SMOTE menghasilkan token sequence sintetis melalui interpolasi k-NN yang sering kali menghasilkan indeks token non-semantik dalam ruang diskrit, sehingga performanya setara dengan baseline natural.

### C. Dampak Task-Adaptive Pretraining (TAPT)
- **Adaptasi Jargon & Slang Kebencanaan**: Tahap TAPT (Masked Language Modeling 3 epoch pada korpus tweet banjir) memberikan peningkatan signifikan pada pemahaman kosakata lokal (misal nama sungai, istilah daerah, singkatan penanganan darurat).
- **Hasil Akhir**: TAPT IndoBERTweet-LoRA menjadi model terbaik mutlak dalam penelitian ini, menembus **Test Accuracy 79.48%** dan **Macro F1 74.92%**, serta mendongkrak Recall kelas Netral hingga **61.20%**.

---

## 3. Kesimpulan Akhir
Seluruh 8 tahapan eksperimen berhasil direplikasi dan dibuktikan secara empiris. Model **TAPT IndoBERTweet-LoRA** direkomendasikan sebagai arsitektur final untuk dilaporkan pada Bab IV dan Bab V Naskah Tesis.
