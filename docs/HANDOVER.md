# HANDOVER — Evolusi Proyek, Deliverables & Cara Pakai

Dokumen serah terima untuk siapa pun yang melanjutkan proyek tesis **Klasifikasi Sentimen Tweet Banjir (LSTM / BiLSTM / IndoBERTweet-LoRA)**. Baca dokumen ini berurutan; indeks dokumen lain ada di bagian akhir.

---

## 1. Ringkasan proyek (1 paragraf)

Proyek membandingkan tiga arsitektur klasifikasi sentimen 3 kelas (negatif/netral/positif) pada tweet banjir (~8.648 data, distribusi 56/27/17): **LSTM**, **BiLSTM**, dan **IndoBERTweet-LoRA** (fine-tuning parameter-efisien). Selain perbandingan model, penelitian bergerak ke arah **calibration-aware classification** (Label Smoothing, Adaptive Smoothing, Threshold Calibration) untuk menaikkan kelas netral yang ambigu — dengan rasio kelas yang moderat sehingga resampling (SMOTE/oversampling) bukan solusi utama.

---

## 2. Evolusi kerja (kronologi)

| Fase | Apa yang dikerjakan | Jejak di repo |
|---|---|---|
| **F0 — Data & preprocessing** | Scraping tweet banjir, pembersihan, ekspansi emoticon, split stratified 80:20 (`split_data.pkl`) | `Data/`, `01_preprocessing.ipynb`, `eman_prepo.ipynb`, `Preprocessing/` |
| **F1 — Pipeline kualitas data** | Audit duplikat/noise/stopword kritis/negasi, anotasi LLM (Gemini), Cohen's Kappa, label-flip analysis | `quality_pipeline/phase0–6`, `reports/`, `Annotation/`, `docs/PRD_DATA_QUALITY_PIPELINE.md`, `docs/GUIDE_DATA_QUALITY_PIPELINE.md` |
| **F2 — Modeling** | Grid search LSTM (120 eksperimen × 5 strategi imbalance), BiLSTM, IndoBERTweet-LoRA (6 trial tuning + simulasi rasio + class weight) | `02_model_lstm.ipynb`, `03_model_bilstm.ipynb`, `04_model_indobertweet_lora.ipynb`, `results/`, `percobaan_dengan_bilstm (2).ipynb` |
| **F3 — P0 Verifikasi evaluasi** | Diagnosis inkonsistensi 0.2393 vs 0.70 → root cause + bug fix + harness verifikasi + sanity check + dokumentasi + siap re-run Kaggle | Lihat deliverables di §3; detail di `docs/P0_VERIFIKASI_EVALUASI.md` |
| **F4 — Re-run Kaggle v5 (hasil terbaru)** *(sesi terkini)* | Kernel `thesis-indobertweet-lora-v1` v5 selesai COMPLETE (30/08). Notebook hasil di `results/thesis-indobertweet-lora-v1.ipynb` (69 sel tereksekusi penuh, memakai `text_bert` + `/kaggle/input`). Hasil tuning 6 trial, retrain trial-4, skenario 1:1:1/6:3:1/8:1:1, dan class weight. Angka final lihat §2.1 di bawah. | `results/thesis-indobertweet-lora-v1.ipynb`, `.kaggle-outputs/`, `temp_kernel_lora/` |

### Temuan kunci F3 (penting untuk memahami angka-angka lama)

- **0.2393 ≠ bug metrik LoRA.** Nilai itu dari grid search **LSTM yang collapse** (selalu menebak kelas mayoritas; rumus: macro F1 = (2p/(1+p))/3, p=0.56 → 0.239). Best config LSTM tidak bermakna sampai training-nya diperbaiki.
- **Angka LoRA lama (acc 0.7797 / Macro F1 0.6994) ternyata dievaluasi di kolom teks yang salah** (`clean_text`, representasi LSTM) akibat silent fallback. Kolom kanonik BERT: **`text_bert`**.
- **Angka kanonik LoRA** (notebook `04`): **acc 0.7954, Macro F1 0.7328** (netral 0.62/0.50/0.55 = bottleneck utama).
- Dua bug nyata di notebook LoRA Kaggle (v1) sudah diperbaiki: sel-29 memakai trainer salah (trial terburuk) dan fallback kolom.

### Temuan kunci F4 (hasil re-run v5 — angka terbaru)

Sumber: `results/thesis-indobertweet-lora-v1.ipynb` (69 sel, run COMPLETE 30/08). Semua angka dari test set n=1730.

**1. Tuning 6 trial (validasi, diurutkan dari terbaik):**

| Trial | batch_size | dropout | lr | r | α | Acc val | Prec macro | Rec macro | **F1 macro val** |
|---|---|---|---|---|---|---|---|---|---|
| **4** | 16 | 0.3 | 2e-4 | 16 | 32 | 0.7211 | 0.6462 | 0.6280 | **0.6326** |
| 2 | 16 | 0.2 | 1e-4 | 8 | 16 | 0.7110 | 0.6265 | 0.5965 | 0.5878 |
| 5 | 32 | 0.2 | 1e-4 | 16 | 32 | 0.7095 | 0.6427 | 0.5714 | 0.5341 |
| 3 | 16 | 0.3 | 1e-4 | 16 | 32 | 0.6922 | 0.5184 | 0.5494 | 0.5036 |
| 1 | 16 | 0.2 | 5e-5 | 8 | 16 | 0.6416 | 0.4139 | 0.5126 | 0.4559 |
| 6 | 32 | 0.3 | 1e-4 | 16 | 32 | 0.6329 | 0.4134 | 0.5194 | 0.4531 |

Trial **4** menang di semua metrik → `best_params` hardcode di sel-22/64 sudah sesuai hasil terbaik.

**2. Retrain trial-4 (empiris, tanpa balancing) — test:**
- Accuracy **0.7254** · Precision macro 0.6519 · Recall macro 0.6338 · **Macro F1 0.6360**
- Per kelas (P/R/F1): negatif 0.8055/0.8421/0.8234 · netral 0.4948/0.3242/0.3918 · positif 0.6552/0.7350/0.6928
- Distribusi prediksi: 1013/192/525 (aktual 969/293/468) → netral tetap bottleneck.

**3. Skenario simulasi (ablation balancing):**

| Skenario | Accuracy | Prec macro | Rec macro | Macro F1 |
|---|---|---|---|---|
| 1:1:1 | 0.6341 | 0.5896 | 0.6189 | 0.5908 |
| 6:3:1 | 0.6931 | 0.4472 | 0.5454 | 0.4911 |
| 8:1:1 | 0.6711 | 0.6134 | 0.4882 | 0.4591 |

Semua skenario **kalah** dari baseline empiris 0.636 → konsisten dengan temuan F2 bahwa balancing tidak membantu.

**4. Class weight (test):**
- Accuracy **0.7145** · Precision macro 0.6363 · Recall macro 0.6329 · **Macro F1 0.6326** — hampir sama dengan empiris (0.636), tidak lebih baik.

> ⚠️ **Catatan konsistensi:** tabel perbandingan di sel-68 (`perbandingan_class_weight.csv`) memuat angka berbeda — LoRA tanpa CW 0.733 / dengan CW 0.735 — yang **tidak cocok** dengan hasil sel-28 (0.636) dan sel-67 (0.633) di notebook yang sama. Jangan pakai angka sel-68 untuk laporan sebelum diverifikasi asal-usulnya.

---

## 3. Deliverables & cara akses

| # | Deliverable | Lokasi | Cara pakai |
|---|---|---|---|
| 1 | **Harness verifikasi metrik** (deteksi collapse, label mapping salah, baseline mayoritas) | `quality_pipeline/verify_metrics.py` | `python -m quality_pipeline.verify_metrics --preds <csv> --y-true label_aktual --y-pred label_prediksi` → laporan ke `reports/verifikasi_evaluasi.md` |
| 2 | **Notebook LoRA v1 yang diperbaiki + dilinearisasi** (bisa di-run linear dari awal) | `results/thesis-indobertweet-lora-v1.ipynb` | Sudah final; jangan edit manual tanpa membaca `P0_VERIFIKASI_EVALUASI.md` §4 |
| 3 | **Salinan siap-push Kaggle** (notebook bersih + metadata kernel) | `temp_kernel_lora/` | `kaggle kernels push -p temp_kernel_lora` |
| 4 | **Sanity check collapse di grid search LSTM** (per trial: distribusi prediksi + baseline + status COLLAPSE/OK) | `02_model_lstm.ipynb`, `results/thesis-lstm-v2.ipynb`, `temp_kernel_lstm/` | Otomatis tercetak saat grid search dijalankan ulang |
| 5 | **Dokumen root cause P0** (0.2393 vs 0.70, angka kanonik, checklist) | `docs/P0_VERIFIKASI_EVALUASI.md` | Referensi utama memahami semua angka lama vs baru |
| 6 | **Rencana pasca re-run + roadmap E1–E3** (gate keputusan per eksperimen) | `docs/PLAN_PASCA_RERUN.md` | Panduan kerja setelah hasil Kaggle ditarik |
| 7 | **Pipeline kualitas data 6 fase** (audit → anotasi LLM → QA → evaluasi) | `quality_pipeline/phase0–6` + `run_all.py` | `python run_all.py` (lihat `docs/GUIDE_DATA_QUALITY_PIPELINE.md`) |
| 8 | **Data & split** | `Data/split_data.pkl` (utama), `data_preprocessed_with_emoticon.csv` | Dimuat langsung oleh notebook model; versi Kaggle di `kaggle_dataset/` |

---

## 4. Status sekarang & agenda terbuka

**Status terbaru (30/08):** re-run Kaggle `thesis-indobertweet-lora-v1` v5 sudah **COMPLETE** dan hasilnya ditarik (angka final lihat §2.1). Langkah berikutnya: jalankan **Fase V1** di `docs/PLAN_PASCA_RERUN.md` (gate verifikasi) → angka final dikunci.

**Agenda berikutnya (urut prioritas):**
1. Fase V1: verifikasi hasil re-run (gate: |Δ Macro F1| ≤ 0.02 vs kanonik 0.7328 — catatan: hasil v5 saat ini 0.636, selisih 0.097, perlu diagnosa dulu).
2. Fase V2: analisis delta per-kelas + distribusi prediksi (netral tetap bottleneck).
3. E1 Label Smoothing → E2 Adaptive LS → E3 Threshold Calibration (gate: acc ≥ 0.80, netral recall ≥ 0.60, Macro F1 ≥ 0.76; ideal Macro F1 ≥ 0.80).
4. Perbaikan training LSTM yang collapse (baseline pembanding di tesis belum jujur sampai ini selesai).
5. Imbalance (class-weight ablation) hanya jika netral recall < 0.55 setelah E1–E3 — bukti eksperimen yang ada: semua perlakuan balancing kalah dari baseline 0.733.

---

## 5. Quick-start untuk penerima proyek (Windows)

```cmd
:: 0. Persiapan
cd /d "D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT"
pip install -r requirements.txt

:: 1. Verifikasi hasil prediksi mana pun (tanpa GPU)
python -m quality_pipeline.verify_metrics --preds hasil_prediksi_test_indobertweet_lora.csv --y-true label_aktual --y-pred label_prediksi

:: 2. Push + pantau training di Kaggle (GPU di cloud, bukan lokal)
kaggle kernels push -p temp_kernel_lora
kaggle kernels status emanuelembuaijdak/thesis-indobertweet-lora-v1
kaggle kernels output emanuelembuaijdak/thesis-indobertweet-lora-v1 -p .kaggle-outputs\

:: 3. Baca hasil run
::    .kaggle-outputs\thesis-indobertweet-lora-v1.ipynb  (notebook + seluruh output)
::    lalu jalankan ulang harness di atas pada CSV hasil pull
```

Prasyarat Kaggle: kredensial di `%USERPROFILE%\.kaggle\kaggle.json` (detail lengkap: `docs/GUIDE_KAGGLE_CLI.md`).

---

## 6. Indeks dokumentasi

| Dokumen | Isi |
|---|---|
| `docs/P0_VERIFIKASI_EVALUASI.md` | Root cause 0.2393 vs 0.70, angka kanonik, checklist P0, troubleshooting evaluasi |
| `docs/PLAN_PASCA_RERUN.md` | Fase V1/V2 + roadmap E1–E3 dengan gate keputusan |
| `docs/GUIDE_KAGGLE_CLI.md` | Workflow Kaggle CLI (push/status/output/pull) |
| `docs/GUIDE_DATA_QUALITY_PIPELINE.md` | Panduan pipeline kualitas data |
| `docs/PRD_DATA_QUALITY_PIPELINE.md` | PRD pipeline kualitas data |
| `docs/DATA_FLOW.md`, `docs/DATA_STRUCTURE.md`, `docs/PREPROCESSING.md`, `docs/MODELS.md` | Referensi data & model |
| `reports/data_quality_audit.md`, `reports/preprocessing_audit.md` | Hasil audit data |

---

## 7. Prinsip kerja yang harus dipertahankan

1. **Tidak ada angka yang dipakai untuk keputusan sebelum lolos verifikasi** (harness + konsistensi antar-sel).
2. **Kolom kanonik BERT = `text_bert`** — jangan pernah fallback diam-diam ke kolom lain.
3. **Evaluasi final selalu dari `trainer_best`**, bukan trainer sisa loop tuning.
4. Sanity check collapse wajib ada di setiap grid search/training loop.
5. Threshold tuning di **validation**, bukan test.
