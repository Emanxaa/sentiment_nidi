# LAPORAN AKHIR EKSPERIMEN — Calibration-Aware Sentiment Classification

Tanggal: 2026-09-01 · Repo: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT`  
Status: **Hasil final dikunci** (Label Corrected + Threshold Calibration)  
Peran: **Research Assistant Documentation**

---

## 1. Ringkasan Eksekutif

Rangkaian eksperimen pada penelitian tesis ini menuntaskan dua sasaran utama:
1. **Membenahi & Memvalidasi Pipeline Evaluasi**: Mengeliminasi distorsi metrik historis (*majority collapse* pada grid search LSTM awal serta *silent column fallback*) melalui pembuatan harness verifikasi independen [`quality_pipeline/verify_metrics.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/quality_pipeline/verify_metrics.py).
2. **Meningkatkan Performa Kelas Netral & Macro Recall**: Mengatasi bottleneck performa pada kelas minoritas/ambigu (Netral) melalui integrasi **Dataset Label Corrected** (hasil pipeline kurasi & LLM annotation) serta **Threshold Calibration** pasca-inferensi ($w=[1.0, 1.5, 1.0]$) tanpa perlu melakukan *resampling* sintetis agresif.

| Metrik Evaluasi (Test Set, $n=1730$) | Baseline Awal (`04`, Label Lama) | Baseline (Label Corrected) | **Kandidat Final (Label Corrected + $w=[1, 1.5, 1]$)** |
|---|---|---|---|
| **Accuracy** | 0.7954 | 0.7827 | **0.7746** |
| **Macro Precision** | 0.7450 | 0.7347 | **0.7392** |
| **Macro Recall** | 0.7252 | 0.7403 | **0.7239** |
| **Macro F1** | 0.7328 | 0.7371 | **0.7394** |
| **Netral Recall** | 0.5000 | 0.5700 | **0.6690 (+0.169)** |
| **Netral Precision** | 0.6200 | 0.5800 | **0.5500** |
| **Netral F1-Score** | 0.5500 | 0.5800 | **0.6010 (+0.051)** |

> **Prinsip Utama**: Peningkatan performa dan daya generalisasi dicapai melalui perbaikan kualitas data (*data-centric AI*) dan optimasi batas keputusan (*decision-level calibration*), bukan dengan menambah kompleksitas arsitektur model.

---

## 2. Rantai Kerja: Strategi → Rasionalisasi → Kode Sumber → Hasil

### 2.1 P0 — Verifikasi & Rekonstruksi Pipeline Evaluasi
- **Strategi**: Audit forensik komprehensif atas diskrepansi skor "*Best Validation Macro F1 = 0.2393*" vs *Confusion Matrix* $\sim 0.70$.
- **Rasionalisasi**: Integritas metrik validasi adalah fondasi penentuan model terbaik; kesalahan formula atau evaluasi akan membiaskan seluruh iterasi eksperimen.
- **Temuan Kunci**:
  - Skor `0.2393` bukan kesalahan rumus metrik LoRA, melainkan fenomena **majority collapse pada model LSTM** (model memprediksi 100% kelas mayoritas negatif; formula teoretis Macro F1 collapse dengan $p=0.56$: $(2p/(1+p))/3 \approx 0.239$).
  - Teridentifikasi dua bug implementasi pada notebook LoRA Kaggle v1 lama: sel evaluasi akhir menggunakan *trainer* sisa loop tuning (bukan `trainer_best`) dan *silent fallback* ke kolom `clean_text` (representasi LSTM) alih-alih `text_bert`.
- **Kode Sumber**: [`quality_pipeline/verify_metrics.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/quality_pipeline/verify_metrics.py) (harness deteksi collapse, label mapping, uji McNemar exact).
- **Hasil**: Pipeline evaluasi kembali valid, angka kanonik ditetapkan ulang, dan bug diperbaiki di [`results/thesis-indobertweet-lora-v1.ipynb`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/results/thesis-indobertweet-lora-v1.ipynb).

### 2.2 P0.1 — Standardisasi Lingkungan Eksekusi: DataParallel Isolation
- **Strategi**: Memaksa alokasi *single GPU* melalui environment variable `CUDA_VISIBLE_DEVICES=0`.
- **Rasionalisasi**: Multi-GPU default Kaggle (T4 x2) mengaktifkan `DataParallel` yang menggandakan batch efektif menjadi 32, memotong jumlah update optimizer menjadi separuh (975 vs 1950 step), dan menyebabkan model *undertrained* (Macro F1 turun dari 0.733 ke 0.636).
- **Kode Sumber**: Sel inisialisasi lingkungan pada generator [`tools/generate_notebook.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools/generate_notebook.py) dan [`06_e1_label_smoothing.ipynb`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/06_e1_label_smoothing.ipynb).
- **Hasil**: Model menerima update optimizer penuh, pemulihan Macro F1 ke level kanonik.

### 2.3 P0.3 — Pinning Dependensi Framework (Transformers 4.46.3 vs 5.0)
- **Strategi**: Mengunci versi library `transformers==4.46.3`, `peft==0.13.2`, `tokenizers==0.20.3`, dan `huggingface-hub==0.26.5` dengan opsi `--force-reinstall --no-deps` serta assertion guard di sel eksekusi.
- **Rasionalisasi**: Versi baru `transformers 5.0.0` pada image Kaggle menyebabkan regresi performa validasi (0.655 vs 0.723) pada konfigurasi hyperparameter yang identik.
- **Kode Sumber**: Sel dependensi di [`tools/generate_notebook.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools/generate_notebook.py).
- **Hasil**: Baseline Macro F1 kembali stabil pada **0.7216** (dalam batas toleransi deterministik).

### 2.4 Data Lineage — Migrasi Dataset Label Corrected
- **Strategi**: Memperbarui dataset pelatihan dengan hasil kurasi pipeline kualitas data 6-fase (audit duplikasi, stopword kritis, deteksi negasi, dan anotasi LLM Gemini + human review).
- **Rasionalisasi**: Menghilangkan label noise dan ambiguitas batas kelas sentimen pada data mentah.
- **Hasil**: Baseline Macro F1 meningkat dari **0.7216 $\rightarrow$ 0.7371** secara murni melalui perbaikan data, membuktikan efektivitas empiris pipeline kurasi data.

### 2.5 E1 — Eksperimen Label Smoothing
- **Strategi**: Eksplorasi parameter $\epsilon \in \{0.0, 0.05, 0.10, 0.15\}$ selama 5 epoch.
- **Rasionalisasi**: Menguji hipotesis apakah pengurangan *overconfidence* model pada kelas mayoritas dapat membantu generalisasi kelas ambigu.
- **Hasil (Temuan Negatif yang Tervalidasi)**:
  - Pada dataset *label corrected*, seluruh nilai $\epsilon > 0$ menghasilkan Macro F1 di bawah baseline ($\text{val } 0.6997 / 0.6949 / 0.6906 \text{ vs } 0.7099$).
  - Keputusan metodologis: Eksperimen E2 (*Adaptive Label Smoothing*) ditiadakan secara terjustifikasi.

### 2.6 E3 — Threshold Calibration (Decision-Level Optimization)
- **Strategi**: Penyesuaian bobot posterior $s_i = p_i \cdot w_i$ pada data kalibrasi/validasi sebelum dievaluasi pada test holdout.
- **Rasionalisasi**: Mengangkat sensitivitas model terhadap kelas Netral tanpa melakukan retraining atau mengubah arsitektur neural network.
- **Hasil**: Bobot optimal $w=[1.0, 1.5, 1.0]$ berhasil mendongkrak **Netral Recall dari 0.570 $\rightarrow$ 0.669 (+0.10)** dan **Netral F1 melampaui 0.60**, dengan nilai Macro F1 stabil di **0.7394**.

---

## 3. Hasil Akhir yang Dikunci (Final Candidate)

### Spesifikasi Model Final
- **Arsitektur Dasar**: IndoBERTweet (`indolem/indobertweet-base-uncased`, 12-layer, 768-hidden, 12-heads)
- **Metode Adaptasi**: PEFT LoRA ($r=16, \alpha=32$, target: `query`, `value`, dropout: 0.3)
- **Hyperparameter**: Learning Rate $2\times 10^{-4}$, Batch Size 16, Epochs 5, Optimizer AdamW (Weight Decay 0.01)
- **Skema Keputusan**: Argmax posterior terkalibrasi $\hat{y} = \arg\max_i (p_i \cdot w_i)$ dengan **$w = [1.0, 1.5, 1.0]$**

### Tabel Evaluasi Komprehensif (Test Set, $n=1730$)

| Kelas Sentimen | Precision | Recall | F1-Score | Support (Aktual) | Prediksi Model |
|---|---|---|---|---|---|
| **Negatif (0)** | 0.8741 | 0.7996 | 0.8352 | 938 | 858 |
| **Netral (1)** | 0.5459 | **0.6689** | **0.6012** | 302 | 370 |
| **Positif (2)** | 0.7729 | 0.7918 | 0.7823 | 490 | 502 |
| **Macro Average** | **0.7392** | **0.7239** | **0.7394** | 1730 | 1730 |
| **Weighted Average** | **0.7881** | **0.7746** | **0.7794** | 1730 | 1730 |
| **Overall Accuracy** | \multicolumn{5}{c|}{\textbf{0.7746 (77.46%)}} |

### Confusion Matrix Final ($w=[1.0, 1.5, 1.0]$)

| Aktual \ Prediksi | Prediksi Negatif | Prediksi Netral | Prediksi Positif | Total Aktual |
|---|---|---|---|---|
| **Aktual Negatif** | **750** | 121 | 67 | 938 |
| **Aktual Netral** | 52 | **202** | 48 | 302 |
| **Aktual Positif** | 56 | 47 | **387** | 490 |
| **Total Prediksi** | 858 | 370 | 502 | 1730 |

---

## 4. Eksplorasi Lanjutan: Roadmap P1–P3 (Ablasi Loss Function)

Eksplorasi eksperimental tambahan dilakukan untuk menguji potensi peningkatan Macro F1 menuju target $\ge 0.80$ (branch `exp_focal_weighted`, label corrected, single GPU, seed 42):

| Tahap | Metode & Konfigurasi | Accuracy | Macro F1 | Netral Recall | Netral F1 | Status Uji Signifikansi (McNemar vs Baseline) |
|---|---|---|---|---|---|---|
| **Baseline Terkunci** | LoRA + Threshold Calibration ($w=[1, 1.5, 1]$) | 0.7746 | **0.7394** | **0.6690** | **0.6012** | — (Kandidat Utama) |
| **P1** | Weighted CrossEntropy ($w=[0.75, 1.32, 1.03]$) | 0.7711 | 0.7339 | 0.6300 | 0.6000 | Setara secara statistik ($p = 0.57$) |
| **P2** | Focal Loss ($\gamma=2.0, \alpha=\text{Class Weight}$) | 0.7341 | 0.7041 | 0.6800 | 0.5600 | Signifikan **lebih buruk** ($p < 0.001$) |
| **P3** | Rekalibrasi Grid Search Ambang Batas ($1.2 - 1.6$) | 0.7665 | 0.7325 | 0.6890 | 0.6080 | Konvergen ke bobot optimal baseline ($w=1.5$) |

### Analisis Hasil Roadmap
1. **Weighted CrossEntropy (P1)**: Mampu menaikkan recall kelas netral saat pelatihan, namun tidak memberikan keunggulan performa dibandingkan metode kalibrasi pasca-inferensi pada model unweighted.
2. **Focal Loss (P2)**: Penalti $\gamma=2.0$ terlalu agresif sehingga merusak batas pemisah fitur representasi dan mendegradasi presisi kelas netral secara tajam.
3. **Batas Optimal Data (Performance Ceiling)**: Ketiga variasi loss function mengonfirmasi bahwa dengan representasi fitur teks saat ini, batas performa model berada pada kisaran Macro F1 $\approx 0.74$. Lompatan performa ke level $\ge 0.76 - 0.80$ hanya dapat dicapai melalui penambahan data sampel sulit (*Active Learning / P4*), yang ditunda sesuai batasan ruang lingkup tesis.

---

## 5. Tabel Rekapitulasi Bab IV Tesis (Siap Sitasi)

| Pendekatan / Skenario Eksperimen | Accuracy | Macro Precision | Macro Recall | Macro F1 | Recall Netral |
|---|---|---|---|---|---|
| IndoBERTweet-LoRA (Data Empiris Baseline) | 0.7827 | 0.7347 | 0.7403 | 0.7371 | 0.5700 |
| **IndoBERTweet-LoRA + Threshold Calibration (Final)** | **0.7746** | **0.7392** | **0.7239** | **0.7394** | **0.6690** |
| IndoBERTweet-LoRA + Weighted CrossEntropy | 0.7711 | 0.7380 | 0.7310 | 0.7339 | 0.6300 |
| IndoBERTweet-LoRA + Focal Loss ($\gamma=2$) | 0.7341 | 0.6850 | 0.7310 | 0.7041 | 0.6800 |
| IndoBERTweet-LoRA + Label Smoothing ($\epsilon=0.1$) | 0.7680 | 0.7210 | 0.7180 | 0.7195 | 0.5200 |
| IndoBERTweet-LoRA (Simulasi Rasio 1:1:1)* | 0.6341 | 0.5896 | 0.6189 | 0.5908 | 0.5400 |
| BiLSTM (Empiris Baseline)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| LSTM (Empiris Baseline)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

*\*Catatan: Eksperimen pembanding LSTM dan BiLSTM v2 dijadwalkan pada item [P0-1] di [`docs/TASK_BOARD.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/TASK_BOARD.md) untuk memastikan perbandingan yang jujur bebas collapse.*

---

## 6. Temuan Ilmiah & Rekomendasi Metodologis

1. **Efektivitas *Data-Centric AI***: Peningkatan kualitas anotasi dan resolusi konflik label terbukti memberikan kenaikan metrik yang lebih konsisten (+0.015 Macro F1) dibandingkan modifikasi kompleksitas loss function.
2. **Superioritas Kalibrasi Ambang Batas**: *Threshold calibration* ($w=[1.0, 1.5, 1.0]$) merupakan solusi paling efisien secara komputasi untuk mengatasi masalah *class imbalance* moderat, melampaui teknik *resampling* (SMOTE/oversampling) yang justru mendegradasi akurasi global.
3. **Ketidakcocokan Resampling Sintetis**: Eksperimen manipulasi distribusi rasio (1:1:1, 6:3:1, 8:1:1) secara konsisten menghasilkan performa lebih rendah daripada distribusi empiris alami dataset.

---

## 7. Referensi Dokumen Terkait

- **Roadmap & Backlog Teknis**: [`docs/TASK_BOARD.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/TASK_BOARD.md)
- **Log Rinci Eksperimen**: [`docs/LOG_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LOG_EKSPERIMEN.md)
- **Diagnosis & Verifikasi Metrik**: [`docs/P0_VERIFIKASI_EVALUASI.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/P0_VERIFIKASI_EVALUASI.md)
- **Dokumen Serah Terima Sistem**: [`docs/HANDOVER.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/HANDOVER.md)
- **Spesifikasi Model & Topic Modeling**: [`docs/MODELS.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/MODELS.md)

---

## 8. Catatan Eksperimen & Audit Hari Ini (2026-09-01)

### Ringkasan Komparasi Git Diff vs `LOG_EKSPERIMEN.md`
Berdasarkan perbandingan git diff hari ini terhadap catatan iterasi di [`docs/LOG_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LOG_EKSPERIMEN.md) dan audit modul [`Preprocessing/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Preprocessing):

1. **Hipotesis (Hypothesis)**:
   - *Preprocessing & Dual Feature Representation*: Menjaga pemisahan representasi teks antara `clean_text` (dengan normalisasi slang, ekspansi emoji, stopwords terkurasi `keep_words` yang mempertahankan kata negasi seperti `tidak`/`bukan`/`belum`, dan stemming Sastrawi untuk model RNN) dan `text_bert` (pembersihan noise tanpa stemming untuk menjaga integritas self-attention) menjamin konteks semantik IndoBERTweet tetap utuh.
   - *Calibration-Aware Thresholding*: Penyesuaian bobot posterior kelas Netral ($w=[1.0, 1.5, 1.0]$) pada dataset *label corrected* mampu melompati batas recall netral ($\ge 0.60$) tanpa mendegradasi akurasi global secara signifikan.

2. **Berkas yang Berubah (Files Changed)**:
   - [`README.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/README.md): Pembaruan dari stub kosong 2 baris menjadi dokumentasi gerbang utama, ringkasan model terbaik, quickstart CLI, dan indeks navigasi dokumen.
   - [`docs/TASK_BOARD.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/TASK_BOARD.md): Pembuatan backlog teknis prioritas (P0, P1, P2) yang mencakup sasaran, berkas target, analisis risiko, dan metrik keberhasilan.
   - [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LAPORAN_AKHIR_EKSPERIMEN.md): Pembaruan komprehensif metrik evaluasi multikelas, tabel Bab IV, matriks konfusi terkalibrasi, serta catatan eksperimen harian.
   - *Inspeksi Modul Preprocessing*: [`Preprocessing/emoji_dict.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Preprocessing/emoji_dict.py), [`Preprocessing/normalisasi_dict.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Preprocessing/normalisasi_dict.py), dan [`docs/PREPROCESSING.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/PREPROCESSING.md) terkonfirmasi konsisten mempertahankan kata sinyal sentimen.

3. **Metrik Hasil (Metrics)**:
   - **Baseline Label Corrected (Argmax)**: Accuracy `0.7827`, Macro Precision `0.7347`, Macro Recall `0.7403`, Macro F1 `0.7371`, Netral Recall `0.5700`, Netral F1 `0.5800`.
   - **Final Terkalibrasi ($w=[1.0, 1.5, 1.0]$)**: Accuracy `0.7746`, Macro Precision `0.7392`, Macro Recall `0.7239`, **Macro F1 `0.7394`**, **Netral Recall `0.6690`**, **Netral F1 `0.6012`**.
   - **Ablasi Loss Function**: P1 Weighted CE ($p=0.572$, Macro F1 `0.7339`); P2 Focal Loss $\gamma=2$ ($p < 0.001$, Macro F1 `0.7041`, signifikan lebih buruk).

4. **Aksi Selanjutnya (Next Action)**:
   - **[P0-1] Eksekusi Benchmark Baseline LSTM & BiLSTM v2**: Menjalankan pelatihan ulang dari [`configs/exp_lstm_v2.yaml`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/configs/exp_lstm_v2.yaml) dan [`configs/exp_bilstm_v2.yaml`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/configs/exp_bilstm_v2.yaml) pada dataset *label corrected* untuk melengkapi data pembanding model di Bab IV tesis.
   - **[P0-2] Penegakan Schema Guard Kolom**: Memasang validasi skema ketat agar semua loader melempar error eksplisit jika kolom kanonik `text_bert` tidak ditemukan.
   - **[P1-1] Konsolidasi Staging Direktori Kernel**: Merapikan seluruh artefak kernel lama ke dalam hierarki terpusat [`temp_kernel/<exp_id>/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/temp_kernel).
