# PRD — Data Quality Improvement Pipeline
**Project:** Thesis-LSTM-IndoBERT
**Version:** 1.0
**Owner:** Emanuel

## Executive Summary
Dokumen ini menjadi panduan operasional untuk meningkatkan kualitas dataset sebelum preprocessing dan pelatihan model LSTM, BiLSTM, dan IndoBERTweet-LoRA. Fokus utama adalah memastikan hasil scraping valid, preprocessing tidak merusak informasi, dan anotasi tervalidasi menggunakan LLM sehingga terbentuk **Gold Dataset v2**.

## Success Metrics
| Metrik | Target |
|---|---:|
| Missing value | <1% |
| Duplicate | <5% |
| Noise teks | <3% |
| Agreement label | >85% |
| Cohen's Kappa | >0.75 |
| Human review error | <10% |

## Existing Pipeline (dipertahankan)
```
data_banjir.csv
      ↓
01_preprocessing.ipynb
      ↓
data_preprocessed_with_emoticon.csv
      ↓
split_data.pkl
      ↓
LSTM / BiLSTM / IndoBERT-LoRA
```

## Enhanced Pipeline (baru)
```
data_banjir.csv
      ↓
Phase 0 — Data Quality Audit
      ↓
LLM Re-Annotation (1000 Sampel)
      ↓
data_banjir_v2.csv
      ↓
01_preprocessing.ipynb
      ↓
split_data.pkl
      ↓
LSTM / BiLSTM / IndoBERT-LoRA
```

## Deliverables
- reports/data_quality_audit.md
- Annotation/gold_dataset_1000.csv
- Annotation/results/*.csv
- Data/data_banjir_v2.csv
- Laporan perbandingan performa sebelum vs sesudah.

# Phase 0 — Data Quality Audit
## Tujuan
Memastikan dataset hasil scraping sudah layak sebelum preprocessing.

### Task 0.1 — Duplicate Audit
- Exact duplicate
- Near duplicate
- Repost
- Output: duplicate_report.csv

### Task 0.2 — Missing Value Audit
Periksa:
- text
- sentimen
- emoticon
- keyword
- created_at

Output:
- missing_report.csv

### Task 0.3 — Noise Detection
Deteksi pola seperti:
- Tampilkan lebih banyak
- 2 rb / 35 rb
- 0:25
- angka engagement
- Dari tribunnews.com

Output:
- noise_pattern_report.csv

### Task 0.4 — Cleaning Rules Baru
Tambahkan regex untuk:
- engagement count
- timestamp video
- sumber berita
- artefak scraping

Contoh:
```
Tampilkan lebih banyak
2 rb
35 rb
0:25
```

# Phase 1 — Preprocessing Audit
## Tujuan
Memastikan preprocessing tidak menghilangkan sinyal sentimen.

### Audit LSTM
Bandingkan:
- text
- text_with_emoticon
- clean_text_lstm

Cek:
- negasi hilang
- sarkasme rusak
- stemming terlalu agresif

### Audit IndoBERT
Bandingkan:
- text_with_emoticon
- text_bert

Pastikan:
- hashtag tetap
- emoji tetap bermakna
- tanda baca penting tidak hilang.

### Stopword Audit
Periksa kata penting:
- tidak
- bukan
- jangan
- sedih
- marah
- doa
- harapan

Output:
- critical_stopword_report.csv

# Phase 2 — Gold Dataset Creation
## Tujuan
Membuat dataset acuan berkualitas tinggi.

### Sampling
Gunakan:
- Stratified Random Sampling
- 1000 tweet
- random_state = 42

Tambahkan kolom:
- id

Output:
- Annotation/gold_dataset_1000.csv

# Phase 3 — LLM Re-Annotation
## Tujuan
Memvalidasi label menggunakan LLM.

### Batch Strategy
- 20 batch
- 50 tweet per batch

### Input
Gunakan:
- text_with_emoticon

Jangan gunakan:
- clean_text_lstm
- processed_text

### Prompt Rules
LLM harus menghasilkan:
```json
{
  "id": 1,
  "label": "negatif",
  "confidence": 98,
  "reason": "Keluhan terhadap penanganan banjir."
}
```

Output setiap batch:
- Annotation/results/batch_001.csv
- ...
- batch_020.csv

# Phase 4 — Quality Assurance
## Tujuan
Menyaring hasil anotasi sebelum menjadi gold dataset.

### Review Rules
Review jika:
- confidence <80
- label berubah dari label lama

### Human Review
Target:
- 100–200 tweet

Output:
- data_banjir_v2.csv

# Phase 5 — Label Quality Evaluation
## Metrik
### Agreement Rate
Hitung persentase kesamaan label lama dan baru.

### Cohen's Kappa
Interpretasi:
| Kappa | Interpretasi |
|---|---|
| <0.40 | Buruk |
| 0.40–0.60 | Sedang |
| 0.60–0.80 | Baik |
| >0.80 | Sangat baik |

### Label Flip Analysis
Kelompokkan perubahan menjadi:
- Sarkasme
- Mixed sentiment
- Informational
- Apresiasi
- Kritik

# Phase 6 — Retraining Pipeline
Setelah dataset selesai:

1. Gunakan data_banjir_v2.csv.
2. Jalankan ulang 01_preprocessing.ipynb.
3. Regenerasi split_data.pkl.
4. Latih ulang:
   - LSTM
   - BiLSTM
   - IndoBERTweet-LoRA
5. Bandingkan Macro F1 sebelum vs sesudah.

# Folder Structure
```
Thesis-LSTM-IndoBERT/
├── docs/
│   ├── DATA_FLOW.md
│   ├── DATA_STRUCTURE.md
│   ├── PREPROCESSING.md
│   ├── MODELS.md
│   └── PRD_DATA_QUALITY.md
│
├── Annotation/
│   ├── batches/
│   ├── results/
│   ├── reports/
│   └── prompts/
│
├── Data/
│   ├── data_banjir.csv
│   ├── data_banjir_v2.csv
│   └── split_data.pkl
```

# Execution Checklist
## Phase 0
- [ ] Audit duplicate.
- [ ] Audit missing value.
- [ ] Audit noise.
- [ ] Dokumentasikan hasil.

## Phase 1
- [ ] Audit preprocessing LSTM.
- [ ] Audit preprocessing IndoBERT.
- [ ] Audit stopword.

## Phase 2
- [ ] Ambil 1000 sampel stratified.
- [ ] Tambahkan ID permanen.

## Phase 3
- [ ] Bagi menjadi 20 batch.
- [ ] Jalankan anotasi LLM.
- [ ] Simpan seluruh batch.

## Phase 4
- [ ] Merge hasil anotasi.
- [ ] Review confidence rendah.
- [ ] Bentuk Gold Dataset v2.

## Phase 5
- [ ] Hitung Agreement Rate.
- [ ] Hitung Cohen's Kappa.
- [ ] Analisis label flip.

## Phase 6
- [ ] Retrain seluruh model.
- [ ] Bandingkan Macro F1.
- [ ] Dokumentasikan perubahan.

# Definition of Done
Pipeline dianggap selesai apabila:
1. Gold Dataset v2 tersedia.
2. Seluruh audit terdokumentasi.
3. Kualitas label terukur.
4. Dataset siap digunakan ulang oleh seluruh pipeline existing.
5. Perbandingan performa sebelum dan sesudah tersedia untuk dimasukkan ke Bab III dan Bab IV tesis.
