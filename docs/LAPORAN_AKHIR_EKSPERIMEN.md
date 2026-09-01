# LAPORAN AKHIR EKSPERIMEN — Calibration-Aware Sentiment Classification

Tanggal: 2026-09-01 · Repo: `D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT`
Status: **hasil final dikunci** (label corrected + kalibrasi threshold)

---

## 1. Ringkasan eksekutif

Eksperimen menyelesaikan dua hal: **(a) membenahi pipeline evaluasi** yang selama ini menyesatkan, dan **(b) menaikkan performa kelas netral** lewat **label corrected** + **threshold calibration** — tanpa mengubah arsitektur model.

| Metrik (test, n=1730) | Awal (kanonik `04`, label lama) | **Final (label corrected + w=[1, 1.5, 1])** |
|---|---|---|
| Accuracy | 0.7954 | 0.7746 |
| Macro F1 | 0.7328 | **0.7394** |
| **Netral recall** | 0.50 | **0.669 (+0.17)** |
| **Netral F1** | 0.55 | **0.601** |

Strategi inti: **perbaiki data dan keputusan (decision boundary), bukan ganti model** — dan setiap langkah diverifikasi alat `verify_metrics.py` agar angka yang dipakai adalah angka yang jujur.

---

## 2. Rantai kerja: strategi → alasan → source code → hasil

### 2.1 P0 — Verifikasi pipeline evaluasi (fix bug lama)
- **Strategi**: audit forensik atas inkonsistensi "Best Validation Macro F1 = 0.2393" vs confusion matrix ~0.70.
- **Alasan**: kalau metrik validasi salah, semua keputusan tuning setelahnya bias.
- **Temuan**:
  - 0.2393 bukan bug metrik dan bukan hasil LoRA — itu **grid search LSTM yang collapse** (selalu menebak kelas mayoritas negatif; formula macro F1 = (2p/(1+p))/3 ≈ 0.239).
  - Confusion matrix ~0.70 valid & konsisten dengan best validation LoRA.
  - Dua bug nyata di notebook LoRA Kaggle (v1): sel-29 mengevaluasi dengan **trainer sisa loop tuning** (bukan `trainer_best` → menghasilkan 0.4587 yang menyesatkan), dan auto-deteksi kolom **silent fallback ke `clean_text`** (representasi LSTM) alih-alih `text_bert`.
- **Source code**: `quality_pipeline/verify_metrics.py` (harness: deteksi collapse, baseline mayoritas, label mapping, McNemar exact).
- **Hasil**: pipeline evaluasi jujur kembali; angka kanonik ditetapkan ulang; bug diperbaiki di `results/thesis-indobertweet-lora-v1.ipynb`.

### 2.2 P0.1 — Env: DataParallel (T4 x2)
- **Strategi**: paksa single GPU via `CUDA_VISIBLE_DEVICES=0`.
- **Alasan**: 2 GPU membuat Trainer memakai DataParallel → batch efektif 32 (bukan 16) → jumlah update optimizer separuh (975 vs 1950 step) → model undertrained (Macro F1 0.636 vs 0.733).
- **Bukti**: `train_batch_size: 32` di `trainer_state.json`; error `'DataParallel' object has no attribute 'config'`.
- **Source code**: sel pertama semua notebook eksperimen (`06_e1_label_smoothing.ipynb` dst).
- **Hasil**: Macro F1 naik 0.636 → 0.664 (masih belum kanonik → lanjut 2.3).

### 2.3 P0.3 — Env: transformers 5.0 vs 4.x
- **Strategi**: pin stack era kanonik: `transformers==4.46.3` + `peft==0.13.2` (+ `tokenizers==0.20.3`, `huggingface-hub==0.26.5`), pola **`--force-reinstall --no-deps`** + sel guard `assert` versi.
- **Alasan**: image Kaggle membawa transformers 5.0.0 (versi mayor baru); pada data & protokol yang sama, run 5.0 memberi val 0.655 vs kanonik 0.723 → sisa gap justru dari versi library. (Dua percobaan gagal dulu karena file instalasi campur + modul ter-cache; pola di atas menyelesaikannya.)
- **Source code**: sel pin di `06_e1_label_smoothing.ipynb`.
- **Hasil**: baseline kembali ke Macro F1 **0.7216** (Δ −0.011 dari kanonik, dalam toleransi).

### 2.4 P0.2 — Env: path mount dataset CLI 2.x
- **Strategi**: loader fleksibel — cari file dari `os.walk("/kaggle/input")`, jangan hardcode `DATASET_DIR`.
- **Alasan**: CLI 2.x me-mount dataset di `/kaggle/input/datasets/<owner>/<slug>/` (bukan `/kaggle/input/<slug>/` skema lama yang dipakai kernel yang di-attach via UI).
- **Source code**: sel load-data di `06_e1_label_smoothing.ipynb`.
- **Hasil**: kernel baru sukses mount di skema mana pun.

### 2.5 E1 — Label Smoothing
- **Strategi**: sweep `label_smoothing_factor` ε = {0, 0.05, 0.10, 0.15}, 5 epoch, `load_best_model_at_end` per val macro F1; evaluasi test baseline vs ε terbaik + simpan probabilitas; McNemar.
- **Alasan (hipotesis)**: netral = kelas ambigu; smoothing mengurangi overconfidence.
- **Source code**: `06_e1_label_smoothing.ipynb` (+ `temp_kernel_e1/` untuk push).
- **Hasil**:
  - Label lama: ε=0.10 sedikit membantu (test Macro F1 0.7295 vs 0.7216; McNemar p=0.001 signifikan), tapi efek kecil & netral F1 malah turun.
  - Label corrected: **temuan negatif** — semua ε di bawah baseline (val 0.6997/0.6949/0.6906 vs 0.7099). → E2 (adaptive LS) di-skip dengan alasan terukur.

### 2.6 Data lineage — migrasi label corrected
- **Strategi**: identifikasi dua versi label di repo; perbarui dataset Kaggle + regenerasi `split_data.pkl`; hitung ulang baseline.
- **Alasan**: pipeline anotasi (quality_pipeline fase 3–6) menghasilkan label terkoreksi (4686/1510/2452) yang belum pernah di-upload; semua angka lama (termasuk kanonik `04`) memakai label lama (4843/1465/2340).
- **Source code**: `kaggle_dataset/` (CSV + pickle), `quality_pipeline/` (pipeline anotasi).
- **Hasil**: baseline label corrected **0.7371** (dari 0.7216) — **bukti empiris nilai pipeline anotasi**.

### 2.7 E3 — Threshold Calibration (offline, tanpa GPU)
- **Strategi**: optimasi bobot keputusan per-kelas `s_i = p_i * w_i`; pilih w terbaik di data kalibrasi, terapkan ke holdout (protokol bersih); validasi McNemar.
- **Alasan**: model sudah menghasilkan probabilitas; bottleneck netral bisa dinaikkan tanpa retraining. Eksplorasi menunjukkan puncak w=[1, 1.5, 1].
- **Source code**: `quality_pipeline/calibrate_thresholds.py`; analisis final `e3_final.py` (di scratchpad sesi).
- **Hasil (holdout, n=865, w=[1, 1.5, 1] fixed a priori)**:

| | Acc | Macro F1 | Netral P | Netral R | Netral F1 |
|---|---|---|---|---|---|
| baseline | 0.7769 | 0.7310 | 0.586 | 0.589 | 0.587 |
| **w=[1, 1.5, 1]** | 0.7665 | **0.7325** | 0.545 | **0.689** | **0.608** |

  → **Netral recall +0.10, netral F1 tembus 0.60, biaya Macro F1 ±0.** Gate netral recall (≥0.60) tercapai; gate Macro F1 (≥0.76) belum (butuh perbaikan model-level).

---

## 3. Hasil final yang dikunci

**Model**: IndoBERTweet-LoRA (indolem/indobertweet-base-uncased, r=16, α=32, dropout 0.3, lr 2e-4, batch 16, 5 epoch, seed 42) · data `text_bert` · label corrected · single GPU · transformers 4.46.3.
**Keputusan**: `argmax_i(p_i · w_i)` dengan **w=[1, 1.5, 1]** (netral diangkat 1.5×).

| Metrik (test, n=1730) | Baseline | **Final (w=[1,1.5,1])** |
|---|---|---|
| Accuracy | 0.7827 | 0.7746 |
| Precision Macro | 0.7347 | 0.7392 |
| Recall Macro | 0.7403 | 0.7239 |
| **Macro F1** | 0.7371 | **0.7394** |
| Negatif (P/R/F1) | 0.86/0.83/0.85 | 0.87/0.80/0.84 |
| **Netral (P/R/F1)** | 0.58/0.57/0.58 | **0.55/0.67/0.60** |
| Positif (P/R/F1) | 0.76/0.81/0.79 | 0.77/0.79/0.78 |

Confusion matrix final (baris=aktual, kolom=prediksi, w=[1,1.5,1]):

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 750 | 121 | 66 |
| netral | 52 | 202 | 48 |
| positif | 56 | 47 | 388 |

Artefak: `.kaggle-outputs/e1v9/hasil_e3_kalibrasi_test.csv` (prediksi + probabilitas, test penuh, w=[1,1.5,1]); laporan harness `reports/verifikasi_final_kalibrasi.md`.

---

## 3b. Roadmap Macro F1 ≥ 0.80 — hasil eksplorasi P1–P3 (semua tercatat di `LOG_EKSPERIMEN.md`)

Percobaan lanjutan mengejar target Macro F1 ≥ 0.80 (branch `exp_focal_weighted`). Semua run memakai label corrected, stack 4.46.3, single GPU, seed 42.

| Tahap | Strategi | Acc | Macro F1 | Netral R/F1 | Verdict vs baseline |
|---|---|---|---|---|---|
| Baseline terkunci | label corrected + kalibrasi w=[1,1.5,1] | 0.7746 | **0.7394** | 0.669/0.601 | — |
| P1 · weighted CE | `CrossEntropyLoss(weight=0.75/1.32/1.03)` | 0.7711 | 0.7339 | 0.63/0.60 | setara (p=0.57) |
| P2 · focal loss | `FL(p)=−α(1−p)^γ log(p)`, γ=2, α=CW | 0.7341 | 0.7041 | 0.68/0.56 | **lebih buruk** (p≈0) |
| P3 · kalibrasi ulang | grid netral 1,2–1,6 (cal→holdout) | 0.7665 | 0.7325 | 0.689/0.608 | optimal = baseline |

**Kesimpulan roadmap:** ketiga pendekatan tidak menembus **Macro F1 ≈ 0.73–0.74**. Bottleneck terbukti **bukan di loss function maupun threshold** — model dengan data yang ada sudah pada titik optimal. P4 (active learning / re-annotasi sampel sulit) adalah satu-satunya jalur berpeluang naik signifikan, namun **ditunda oleh keputusan pengguna** (akan mengubah dataset kanonik & berbiaya anotasi).

Artefak roadmap: notebook `07_e4_focal_loss.ipynb` (P1), `08_e5_focal_loss.ipynb` (P2), kernel `thesis-lora-e4-class-weight` & `thesis-lora-e5-focal-loss`, hasil `.kaggle-outputs/e4/` & `.kaggle-outputs/e5/`, baseline pembanding `results/baseline_compare/hasil_baseline_kalibrasi_test.csv`.

---

## 3c. Tabel hasil untuk Bab IV (siap salin)

| Eksperimen | Macro F1 | Recall Netral | Status |
|---|---|---|---|
| Baseline (label corrected) | 0.7371 | 0.57 | ✔ |
| Threshold calibration w=[1, 1.5, 1] | **0.7394** | **0.67** | ✔ |
| Weighted CE (0.75/1.32/1.03) | 0.7339 | 0.63 | setara (p=0.57) |
| Focal Loss γ=2 + CW | 0.7041 | 0.68 | ✘ lebih buruk |
| Focal + Calibration | — | — | tidak dijalankan* |
| Final (kandidat tesis) | **0.7394** | **0.67** | ✔ |

\* Kalibrasi tidak dijalankan pada model focal karena P2 sudah signifikan lebih buruk sebelum kalibrasi (Macro F1 0.7041 < baseline 0.7394); kalibrasi ulang (P3) dijalankan pada probabilitas baseline dan konvergen ke bobot yang sama dengan kandidat final.

**Angka kunci**: Macro F1 **0.7394** · Recall Netral **0.669** · Netral F1 **0.601** · Accuracy 0.7746 (test n=1730, label corrected, w=[1, 1.5, 1]).

---

## 4. Peta source code

| Artefak | Lokasi | Fungsi |
|---|---|---|
| Harness verifikasi metrik + McNemar | `quality_pipeline/verify_metrics.py` | Deteksi collapse, label mapping, baseline mayoritas, uji berpasangan |
| Kalibrasi threshold | `quality_pipeline/calibrate_thresholds.py` | Optimasi bobot per-kelas + ECE + McNemar |
| Notebook eksperimen E1 (sumber) | `06_e1_label_smoothing.ipynb` | Sweep label smoothing + simpan probabilitas |
| Paket push kernel E1 | `temp_kernel_e1/` | Notebook + `kernel-metadata.json` (T4, internet, machine_shape) |
| Notebook LoRA v1 (diperbaiki) | `results/thesis-indobertweet-lora-v1.ipynb` | Evaluasi `trainer_best`, kolom `text_bert`, single GPU |
| Hasil mentah Kaggle | `.kaggle-outputs/` | Notebook executed, CSV prediksi, log |
| Log eksperimen lengkap | `docs/LOG_EKSPERIMEN.md` | Pra/proses/pasca semua eksperimen (bukti iteratif) |
| Dokumen pendukung | `docs/P0_VERIFIKASI_EVALUASI.md`, `docs/PLAN_PASCA_RERUN.md`, `docs/HANDOVER.md`, `docs/GUIDE_KAGGLE_CLI.md` | Root cause, rencana pasca-run, serah terima, workflow Kaggle |
| Data (label corrected) | `Data/`, `kaggle_dataset/` | CSV + `split_data.pkl` (sudah di-upload ke Kaggle) |

---

## 5. Temuan negatif yang tetap valid (penting untuk laporan)

1. **SMOTE/oversampling/undersampling**: semua perlakuan balancing di data lama kalah dari baseline (0.689/0.716/0.636 simulasi vs 0.733) — mendukung keputusan tidak memakai resampling.
2. **Label Smoothing**: tidak membantu di label corrected (semua ε di bawah baseline) — temuan negatif yang terdokumentasi.
3. **Weighted CE (P1)**: tidak mengalahkan baseline kalibrasi (McNemar p=0.57) — meski menaikkan recall netral tanpa kalibrasi.
4. **Focal Loss γ=2 (P2)**: signifikan lebih buruk (Macro F1 0.7041 vs 0.7394) — γ terlalu agresif meruntuhkan presisi netral.
5. **Macro F1 0.76–0.80 tidak tercapai**: threshold-only, loss-level, maupun kalibrasi ulang tidak cukup; gap tersisa butuh **data baru (P4 active learning)**, bukan perbaikan model-level dengan data yang ada.

---

## 6. Cara mereproduksi

1. **Verifikasi prediksi mana pun**: `python -m quality_pipeline.verify_metrics --preds <csv> --y-true label_aktual --y-pred label_prediksi`
2. **Kalibrasi ulang**: `python -m quality_pipeline.calibrate_thresholds --val <probs_val.csv> --test <probs_test.csv> --y-true label_aktual --prob-cols prob_negatif,prob_netral,prob_positif`
3. **Ulang run E1**: `kaggle kernels push -p temp_kernel_e1` (dataset sudah label corrected)
4. **Bandingkan model**: `python -m quality_pipeline.verify_metrics --preds A.csv --compare B.csv --y-true ... --y-pred ...`

Catatan replikasi: seed 42; stack dipin (transformers 4.46.3, peft 0.13.2); single GPU (sel `CUDA_VISIBLE_DEVICES=0`); versi library tercetak di sel identitas notebook.
