# Bukti Proyek untuk Pembimbing / Stakeholder
**Analisis Sentimen Bencana Banjir Menggunakan Model IndoBERTweet-LoRA dan Threshold Calibration**

Dokumen ini memuat rangkuman bukti empiris dan artefak repositori berdasarkan 9 fase perjalanan proyek penelitian. Laporan komprehensif naratif untuk naskah Bab III & IV dapat diakses pada [**`docs/LAPORAN_AKHIR_EKSPERIMEN.md`**](docs/LAPORAN_AKHIR_EKSPERIMEN.md).

---

## 1. Bukti Perjalanan Proyek per Fase

### Fase 1 – Membangun Fondasi

* **Struktur Repo yang Sudah Dirapikan**:
  * Struktur repositori telah distandardisasi secara modular:
    * [`configs/`](configs/) : Konfigurasi eksperimen YAML yang terisolasi.
    * [`quality_pipeline/`](quality_pipeline/) : Modul audit, anotasi, kalibrasi, dan verifikasi metrik.
    * [`reports/`](reports/) : Laporan verifikasi markdown & CSV artefak audit.
    * [`docs/`](docs/) : Dokumentasi alur data, log eksperimen, PRD, dan SOP pengerjaan.
    * [`Annotation/`](Annotation/) : Log batch anotasi LLM dan validasi *human review*.
* **Log Eksperimen yang Terdokumentasi**:
  * Terdokumentasi lengkap di [`docs/LOG_EKSPERIMEN.md`](docs/LOG_EKSPERIMEN.md) dengan kontrak standar:
    * **PRA**: Hipotesis, kondisi sebelum push, parameter yang diubah, ekspektasi, dan cara ukur.
    * **PROSES**: Kronologi run di Kaggle, penanganan error, retry, dan diagnosis lingkungan.
    * **PASCA**: Hasil metrik kuantitatif, uji signifikansi statistik McNemar, dan analisis temuan.
* **Versi Environment yang Dikunci**:
  * Dependensi terkunci di [`requirements.txt`](requirements.txt) & [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
  * Stack library dipin ke `transformers==4.46.3`, `tokenizers==0.20.3`, `peft==0.19.1`.
  * Isolasi Single-GPU (`CUDA_VISIBLE_DEVICES=0`) di seluruh kernel untuk mencegah *DataParallel batch-doubling* yang memicu *undertrained model*.
* **Validator Metrik (`verify_metrics.py`)**:
  * Script mandiri di [`quality_pipeline/verify_metrics.py`](quality_pipeline/verify_metrics.py) dengan fitur:
    * Deteksi otomatis *Majority Collapse* ($Macro\ F1 \le 0.239$).
    * Validasi skema label dan distribusi prediksi.
    * Auto-generate laporan evaluasi markdown ke folder [`reports/`](reports/).

---

### Fase 2 – Menetapkan Baseline

* **Notebook Baseline**:
  * Notebook kanonik: [`04_model_indobertweet_lora.ipynb`](04_model_indobertweet_lora.ipynb) dan hasil eksekusi Kaggle di [`results/thesis-indobertweet-lora-v1.ipynb`](results/thesis-indobertweet-lora-v1.ipynb).
* **Classification Report Baseline (Data Uji $n = 1.730$)**:
  * Sumber kanonik: [`docs/P0_VERIFIKASI_EVALUASI.md`](docs/P0_VERIFIKASI_EVALUASI.md).

| Kategori Sentimen | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Negatif (0)** | 0.85 | 0.89 | 0.87 | 969 |
| **Netral (1)** | 0.62 | **0.50** | **0.55** | 293 |
| **Positif (2)** | 0.76 | 0.78 | 0.77 | 468 |
| **Macro Average** | **0.7450** | **0.7252** | **0.7328** | **1.730** |

* **Nilai Metrik Baseline Awal**:
  * **Akurasi**: **79.54%**
  * **Macro F1-Score**: **0.7328**
  * **Recall Kelas Netral**: **50.00%** (titik kelemahan utama baseline awal).

---

### Fase 3 – Menguji Apakah Model Menjadi Bottleneck

* **Tabel Hyperparameter Search (Tuning LoRA 6 Trial)**:
  * Sumber data: [`docs/P0_VERIFIKASI_EVALUASI.md`](docs/P0_VERIFIKASI_EVALUASI.md) & [`results/thesis-indobertweet-lora-v1.ipynb`](results/thesis-indobertweet-lora-v1.ipynb).

| Trial | Batch Size | Dropout | Learning Rate | LoRA $r$ | LoRA $\alpha$ | Val Accuracy | Val Macro F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Trial 4 (Best)** | **16** | **0.3** | **2e-4** | **16** | **32** | **0.7211** | **0.6326** (val v5) / **0.7233** (kanonik) |
| Trial 2 | 16 | 0.2 | 1e-4 | 8 | 16 | 0.7110 | 0.5878 |
| Trial 5 | 32 | 0.2 | 1e-4 | 16 | 32 | 0.7095 | 0.5341 |
| Trial 3 | 16 | 0.3 | 1e-4 | 16 | 32 | 0.6922 | 0.5036 |
| Trial 1 | 16 | 0.2 | 5e-5 | 8 | 16 | 0.6416 | 0.4559 |
| Trial 6 | 32 | 0.3 | 1e-4 | 16 | 32 | 0.6329 | 0.4531 |

* **Bukti Empiris**:
  * Perubahan parameter hanya mengoptimalkan stabilitas konvergensi, namun performa model tetap berada di batas *plateau* ($Macro\ F1 \approx 0.73$).
  * Variasi parameter arsitektur/tuning tidak mampu mendongkrak Recall Netral secara mandiri tanpa intervensi data atau kalibrasi keputusan.

---

### Fase 4 – Menguji Kualitas Data

* **Dokumentasi Proses Anotasi Ulang**:
  * Modul eksekusi 6-fase di [`quality_pipeline/`](quality_pipeline/) dan dokumentasi di [`docs/PRD_DATA_QUALITY_PIPELINE.md`](docs/PRD_DATA_QUALITY_PIPELINE.md).
  * Dataset gold audit: [`Annotation/gold_dataset_1000.csv`](Annotation/gold_dataset_1000.csv) dan laporan di [`Annotation/reports/evaluation_report.md`](Annotation/reports/evaluation_report.md).
  * Metrik kesepakatan: **Agreement Rate 65.80%**, **Cohen's Kappa = 0.4609 (Kategori Sedang)**.
* **Contoh Label Sebelum & Sesudah (*Label Flip*)**:
  * Total **342 dari 1.000 tweet (34.2%)** mengalami koreksi label ([`Annotation/reports/label_flip_analysis.csv`](Annotation/reports/label_flip_analysis.csv)):
    * **Negatif $\rightarrow$ Netral (91 tweet)**: Tweet informatif seputar debit air / status posko banjir yang sebelumnya salah dilabeli negatif hanya karena mengandung kata *"banjir"*.
    * **Negatif $\rightarrow$ Positif (120 tweet)**: Tweet doa keselamatan, apresiasi relawan, dan informasi bantuan.
    * **Netral $\rightarrow$ Positif (45 tweet)** & **Netral $\rightarrow$ Negatif (33 tweet)**.
* **Perbandingan Performa Sebelum vs Sesudah Koreksi Data**:
  * Baseline data lama: Macro F1 = **0.7216** (v7) / **0.7328** (kanonik), Recall Netral = **50.00%**.
  * Baseline data terkoreksi ([`reports/verifikasi_e0_labelbaru.md`](reports/verifikasi_e0_labelbaru.md)): Macro F1 naik ke **0.7371**, Recall Netral naik ke **57.28%**, F1 Netral naik ke **0.58**.

---

### Fase 5 – Menguji Hipotesis Imbalance

* **Grafik Distribusi Kelas**:
  * File visualisasi: [`Output/Wordcloud & Distribusi Data/distribusi_sentimen.png`](Output/Wordcloud%20&%20Distribusi%20Data/distribusi_sentimen.png).
  * Sebaran data: Negatif **54.2%** ($n=4.686$), Positif **28.3%** ($n=2.452$), Netral **17.5%** ($n=1.510$).
* **Perbandingan Metrik Eksperimen Penanganan Imbalance ($n=1.730$)**:

| Pendekatan / Eksperimen | Akurasi | Macro F1 | Recall Netral | F1 Netral | Status / Uji Signifikansi |
|---|:---:|:---:|:---:|:---:|---|
| **Class Weighting (P1 / Weighted CE)** | 77.11% | **0.7339** | 63.00% | 0.60 | McNemar $p=0.572$ (tidak berbeda signifikan) |
| **Focal Loss $\gamma=2$ (P2)** | 73.41% | **0.7041** | 68.00% | 0.56 | McNemar $p < 0.0001$ (**Signifikan Lebih Buruk**) |
| **Label Smoothing $\epsilon=0.1$** | 76.80% | **0.7195** | 52.00% | 0.55 | Sinyal pembeda kelas bersih menjadi kabur |
| **Resampling Sintetis / Rasio 1:1:1** | 63.41% | **0.5908** | 54.00% | 0.49 | Sintesis teks merusak distribusi semantik |

---

### Fase 6 – Analisis Kesalahan Model (Error Analysis)

* **Confusion Matrix Model Terbaik (Data Uji $n=1.730$)**:
  * Sumber: [`reports/verifikasi_final_kalibrasi.md`](reports/verifikasi_final_kalibrasi.md).

```
                      PREDIKSI MODEL
                 Negatif   Netral   Positif │  Total Aktual
 ┌─────────────┬─────────┬────────┬─────────┤
 │ Negatif     │   750   │  121   │   66    │   937 tweet
 │ Netral      │    52   │  202   │   48    │   302 tweet
 │ Positif     │    56   │   47   │  388    │   491 tweet
 └─────────────┴─────────┴────────┴─────────┘
```

* **Classification Report Terbaik**:
  * **Negatif**: Precision 0.87 · Recall 0.80 · F1 **0.84** (support 937)
  * **Netral**: Precision 0.55 · Recall **0.67** · F1 **0.60** (support 302)
  * **Positif**: Precision 0.77 · Recall 0.79 · F1 **0.78** (support 491)
  * **Macro Avg**: Precision 0.73 · Recall 0.75 · **Macro F1 = 0.7394** (Akurasi 77.46%)
* **Bukti Kelas Netral Titik Lemah**:
  * Precision kelas netral berada di 0.55 karena tingginya *false positive* dari tweet negatif informatif (121 sampel negatif terprediksi netral).

---

### Fase 7 – Threshold Calibration

* **Perbandingan Sebelum vs Sesudah Kalibrasi ($w = [1.0, 1.5, 1.0]$)**:
  * Script implementasi: [`quality_pipeline/calibrate_thresholds.py`](quality_pipeline/calibrate_thresholds.py).
  * Laporan verifikasi: [`reports/verifikasi_final_kalibrasi.md`](reports/verifikasi_final_kalibrasi.md).

| Metrik Evaluasi | Sebelum Kalibrasi (Argmax) | Setelah Kalibrasi ($w=[1, 1.5, 1]$) | Perubahan ($\Delta$) |
|---|:---:|:---:|:---:|
| **Akurasi** | 78.27% | 77.46% | $-0.81\%$ |
| **Macro F1-Score** | 0.7371 | **0.7394** | **$+0.0023$** |
| **Recall Kelas Netral** | 57.28% (173/302) | **66.89% (202/302)** | **$+9.61\%$** ($+16.9\%$ vs baseline lama) |
| **F1-Score Kelas Netral** | 0.5800 | **0.6012** | **$+0.0212$** (Tembus batas 0.60) |

---

### Fase 8 – Kesimpulan Berbasis Bukti

* **Teknik yang Berhasil**:
  1. **Koreksi Kualitas Label (LLM + Human Review)**: Mengeliminasi bias anotasi dan menaikkan Macro F1 baseline dari 0.7216 ke 0.7371.
  2. **Threshold Calibration ($w=[1.0, 1.5, 1.0]$)**: Mendongkrak Recall Netral hingga 66.89% dan F1 Netral ke 0.6012 secara efisien tanpa retraining.
  3. **Standardisasi Stack & Single-GPU**: Menjamin reproduktifitas eksperimen yang konsisten.
* **Teknik yang Gagal / Negatif**:
  1. **SMOTE & Resampling Teks**: Merusak sintaksis kalimat dan menurunkan Macro F1 ke kisaran 0.45 – 0.59.
  2. **Focal Loss ($\gamma=2$)**: Mengakibatkan *over-penalization* sehingga presisi netral jatuh ke 0.48 (Macro F1 drop signifikan ke 0.7041, $p < 0.001$).
  3. **Label Smoothing**: Mengaburkan sinyal pembeda pada dataset yang sudah bersih.

---

### Fase 9 – Bukti Arah Pengembangan Berikutnya

* **Decision Tree Roadmap**:
  * Terdokumentasi lengkap di [`docs/TASK_BOARD.md`](docs/TASK_BOARD.md).
* **Daftar Eksperimen Prioritas (Target Macro F1 $\ge 0.80$)**:
  1. **P1 — Fine-Tuning Sweep Lanjutan** (✅ Selesai): 10 varian diuji ([`configs/exp_p1_ft_sweep.yaml`](configs/exp_p1_ft_sweep.yaml)). Varian terbaik `t10` (LR 2e-4, Ep 8, Wm 0.1, WD 0.01, Len 64) menghasilkan Akurasi 77.98% / Macro F1 **0.7390** / Recall Netral 61.26%. Membuktikan secara definitif bahwa hyperparameter bukan bottleneck.
  2. **P2 — Task-Adaptive Pretraining (TAPT via MLM 3 Epoch)** (🟡 Berjalan): Adaptasi domain kosakata tweet banjir lokal Sumatera (nama sungai, istilah debit air) pada seluruh korpus 8.648 tweet (`notebooks/exp_p2_tapt_mlm.ipynb`).
  3. **P3 — Representation Benchmark**: Evaluasi komparasi fitur TF-IDF vs Sentence-BERT vs IndoBERTweet.
  4. **P4 — Backbone Swap**: Eksplorasi backbone alternatif (IndoRoBERTa / XLM-RoBERTa).
  5. **P5 — Ensemble Model**: Penggabungan probabilitas model terbaik via *soft-voting*.
* **Alasan Pemilihan**: Eksperimen fungsi loss, sampling, dan hyperparameter sweep (P1) semuanya mengonfirmasi batas maksimal (*performance ceiling* $\approx 0.74$). Peningkatan berikutnya wajib berfokus pada **peningkatan kapasitas representasi domain bahasa**.

---

## 2. Ringkasan Satu Slide (Executive Summary Table)

| Fase | Bukti Utama di Repositori | Output / Metrik Kunci |
|---|---|---|
| **Fondasi** | [`configs/`](configs/), [`requirements.txt`](requirements.txt), [`quality_pipeline/verify_metrics.py`](quality_pipeline/verify_metrics.py) | Repo modular, env terkunci (`transformers 4.46.3`), validator metrik aktif |
| **Baseline** | [`04_model_indobertweet_lora.ipynb`](04_model_indobertweet_lora.ipynb), [`docs/P0_VERIFIKASI_EVALUASI.md`](docs/P0_VERIFIKASI_EVALUASI.md) | Acc 79.54%, Macro F1 **0.7328**, Recall Netral **50.00%** |
| **Hyperparameter** | [`docs/P0_VERIFIKASI_EVALUASI.md`](docs/P0_VERIFIKASI_EVALUASI.md), [`results/thesis-indobertweet-lora-v1.ipynb`](results/thesis-indobertweet-lora-v1.ipynb) | Tuning 6 trial membuktikan plateau pada konfigurasi LoRA ($r=16, \alpha=32$) |
| **Data Quality** | [`Annotation/gold_dataset_1000.csv`](Annotation/gold_dataset_1000.csv), [`Annotation/reports/evaluation_report.md`](Annotation/reports/evaluation_report.md) | 342 label terkoreksi ($\kappa=0.4609$), baseline naik ke Macro F1 **0.7371** |
| **Imbalance** | [`reports/verifikasi_e4_p1.md`](reports/verifikasi_e4_p1.md), [`reports/verifikasi_e5_p2.md`](reports/verifikasi_e5_p2.md) | Focal Loss & Smoothing gagal; SMOTE merusak representasi teks |
| **Analisis Model** | [`reports/verifikasi_final_kalibrasi.md`](reports/verifikasi_final_kalibrasi.md), [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](docs/LAPORAN_AKHIR_EKSPERIMEN.md) | Titik lemah terbukti pada ambiguitas kelas Netral vs Negatif |
| **Calibration** | [`quality_pipeline/calibrate_thresholds.py`](quality_pipeline/calibrate_thresholds.py), [`reports/verifikasi_final_kalibrasi.md`](reports/verifikasi_final_kalibrasi.md) | Recall Netral melonjak **50.0% $\rightarrow$ 66.9%**, Macro F1 naik ke **0.7394** |
| **P1 FT Sweep** | [`reports/verifikasi_p1_ft_sweep.md`](reports/verifikasi_p1_ft_sweep.md), [`configs/exp_p1_ft_sweep.yaml`](configs/exp_p1_ft_sweep.yaml) | Sweep 10 varian capai **0.7390** (Recall Netral 61.3%); bukti batas representasi |
| **Kesimpulan** | [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](docs/LAPORAN_AKHIR_EKSPERIMEN.md) | *Data-centric* & kalibrasi terbukti efektif; manipulasi loss/resampling gagal |
| **Roadmap** | [`docs/TASK_BOARD.md`](docs/TASK_BOARD.md), [`notebooks/exp_p2_tapt_mlm.ipynb`](notebooks/exp_p2_tapt_mlm.ipynb) | **P2 TAPT (MLM 3-Epoch) sedang berjalan** menuju target $\ge 0.80$ |

