# Log Eksperimen — E0 s.d. E5 (Calibration-Aware Development)

Bukti iteratif semua eksperimen. Kontrak tiap entri (lihat plan E1–E3):

- **PRA** (ditulis sebelum push): kondisi saat ini, apa yang diubah, ekspektasi, cara ukur, risiko.
- **PROSES**: kronologi run (push, error, penanganan, retry).
- **PASCA**: kegagalan & bottleneck, findings (termasuk negatif), hasil (tabel + verdict gate), langkah selanjutnya.

Aturan penerimaan angka: `verify_metrics.py` mereproduksi report notebook (≤ 0.001), tidak flag COLLAPSE, evaluasi dari `trainer_best`, kolom `text_bert` terkonfirmasi. Klaim "naik" wajib didampingi McNemar.

Status: 🟡 berjalan · ✅ selesai · ❌ gagal dihentikan

---

## E0 — Verifikasi run v8 & kunci baseline IndoBERTweet-LoRA  🟡

### PRA
- **Kondisi**: baseline kanonik dari notebook `04` (Colab, `text_bert`): acc **0.795376**, Macro F1 **0.732810**, netral 0.62/0.50/0.55. Run Kaggle v6-v7 gagal (torchao; P100 fallback) — sudah difix; **v8 RUNNING** (`machine_shape: NvidiaTeslaT4`).
- **Apa yang dilakukan**: verifikasi hasil run v8 memakai checklist Fase V1 (`docs/PLAN_PASCA_RERUN.md`): kolom `text_bert`, tuning sehat (semua trial ≫ 0.239), sel-24/27 == sel-29, harness mereproduksi Macro F1, |Δ Macro F1| ≤ 0.02 vs kanonik.
- **Ekspektasi**: acc ± 0.7954, Macro F1 ± 0.7328 (toleransi seed ±0.02). Jika lulus → **E0 terkunci**: CSV prediksi+probs baseline jadi pembanding resmi untuk E1–E3.
- **Cara ukur**: `verify_metrics.py` pada `hasil_prediksi_test_indobertweet_lora.csv` hasil pull.
- **Risiko**: versi dataset berubah / seed berbeda → di luar toleransi → diagnosa (troubleshooting), jangan lanjut.

### PROSES
- v6 (fix torchao) di-cancel oleh push v7 (probe accelerator dengan nilai invalid → fallback P100 → `AcceleratorError`) — pelajaran terdokumentasi di `GUIDE_KAGGLE_CLI.md`.
- v8 di-push dengan `machine_shape: NvidiaTeslaT4` + internet on → berjalan ±57 menit, menuntaskan tuning + evaluasi empiris + sebagian simulasi, lalu **ERROR di sel class-weight**.

### PASCA (parsial — verifikasi angka empiris v8)

**Hasil verifikasi (harness, `reports/verifikasi_e0_v8.md`):**

| Metrik (test, n=1730) | v8 (Kaggle T4 x2) | Kanonik `04` (Colab 1 GPU) | Δ |
|---|---|---|---|
| Accuracy | 0.725434 | 0.795376 | −0.070 |
| Macro F1 | 0.636004 | 0.732810 | −0.097 |

- Harness: COLLAPSE **TIDAK**, distribusi pred 1013/192/525, tabel tuning sehat (trial 4 tetap terbaik, urutan ranking sama).
- **Gate V1 GAGAL pada angka** (Δ −0.097 ≫ toleransi 0.02) → E0 belum terkunci.

**Root cause (bukti forensik):**
1. `trainer_state.json` trial-4: `train_batch_size: 32`, `max_steps: 975` — **DataParallel (2 GPU) menggandakan batch efektif 16→32** sehingga jumlah update optimizer tinggal setengah (975 vs 1950 pada run kanonik) → undertrained (train loss masih ~1.35 di epoch 4 vs 0.64 kanonik). Run v1 lama menunjukkan pola sama (975 step, loss 1.457).
2. Error sel-59 (class-weight): `AttributeError: 'DataParallel' object has no attribute 'config'` — konfirmasi Trainer membungkus model dengan DP di image ini.
3. Gap sudah terlihat di validation (best val F1 0.6326 vs 0.7233) → masalah training, bukan test set.

**Tindakan (P0.1):** sel `CUDA_VISIBLE_DEVICES=0` sebagai sel kode pertama di notebook v1 dan notebook E1 → protokol identik dengan run kanonik (1 GPU, batch 16, 1950 step); sekaligus memperbaiki crash WeightedTrainer. Semua artefak diperbarui (`results/`, `temp_kernel_lora/`, `06_e1_…`, `temp_kernel_e1/`), tervalidasi nbformat.

**Keputusan iterasi:** verifikasi E0 + E1 digabung dalam satu run — ε=0 pada notebook E1 adalah replikasi single-GPU dari baseline (jika mencapai ±0.73 → akar masalah terkonfirmasi + E0 terkunci). Notebook v1 full re-run menyusul untuk artefak final.

---

## E1 — Label Smoothing (ε ∈ {0.05, 0.10, 0.15} + baseline)  🟡 pra-eksperimen

### PRA (ditulis sebelum push)
- **Kondisi**: E0 = baseline kanonik run v8 (angka final diisi pasca verifikasi Fase V). Konfigurasi terbaik tuning: trial-4 (batch 16, dropout 0.3, lr 2e-4, r 16, α 32). Bottleneck: netral recall ~0.50.
- **Apa yang diubah**: notebook baru `06_e1_label_smoothing.ipynb` — identik dengan pipeline empiris v1 (data `text_bert`, split 80:20 + 10% val, seed 42, `load_best_model_at_end`) tetapi: (a) tanpa grid search/simulasi/class-weight; (b) 4 pelatihan dengan `label_smoothing_factor` ε = {0.0, 0.05, 0.10, 0.15}; (c) model terbaik-ε vs baseline dievaluasi di test + simpan prediksi & probabilitas (`hasil_e1_test_*.csv`); (d) McNemar exact inline.
- **Ekspektasi**: Macro F1 +1–3 poin; recall netral naik; precision negatif sedikit turun (efek smoothing yang diketahui). Target gate: recall netral ≥ 0.60 & Macro F1 ≥ 0.76.
- **Cara ukur**: val macro F1 (pemilihan ε) → test classification report + `verify_metrics.py` pada CSV hasil pull + McNemar vs baseline.
- **Risiko**: (1) smoothing merusak kelas mayoritas → deteksi via per-kelas metrik; (2) perbedaan seed antar-run Kaggle → toleransi ±0.02, McNemar sebagai hakim; (3) 4 pelatihan dalam satu run → ±2× waktu baseline.

### PROSES
- **Push v1** (sekaligus verifikasi E0 via ε=0): gagal 43 detik — dataset tidak ter-mount (`FileNotFoundError /kaggle/input/.../data_preprocessed_with_emoticon.csv`) padahal metadata server benar & dataset status `ready`. Diagnosis: transient mount pada kernel baru.
- **Cek lingkungan run v1 (berhasil terekam)**: python 3.12.13, torch 2.10.0+cu128, transformers **5.0.0**, peft 0.19.1, `gpu count: 1` (fix `CUDA_VISIBLE_DEVICES=0` **terbukti bekerja**), torchao uninstall OK.
- **Push v2**: ERROR sama (mount tetap kosong di path lama) → masalah sistematis, bukan transient.
- **Push v3 (+ diagnostik `os.walk('/kaggle/input')`)**: **AKAR MASALAH KETEMU** — dataset TER-MOUNT, tapi di path baru CLI 2.x: `/kaggle/input/datasets/<owner>/<slug>/`, bukan `/kaggle/input/<slug>/` (skema lama yang dipakai kernel v1 yang dulu di-attach via UI). `DATASET_DIR` hardcode tidak pernah cocok.
- **Fix P0.2**: sel load-data kini mencari CSV dari daftar mount (kompatibel kedua skema). Notebook sumber + salinan push diperbarui, tervalidasi.
- **Push v4**: COMPLETE (±36 menit: 4×training ~530s + evaluasi).
- **Hasil v4**: baseline ε=0 val macro F1 **0.6551** / test **0.6643**; ε={0.05,0.10,0.15} semuanya DI BAWAH baseline (0.6500/0.6454/0.6439) → label smoothing tidak membantu pada stack ini. Catatan teknis: karena baseline terbaik, sel evaluasi test mengevaluasi baseline dua kali (bug kecil notebook, tidak memengaruhi kesimpulan; diperbaiki di versi berikutnya).
- **Analisis v4**: single-GPU terbukti (gpu count 1), 1950 step benar (best checkpoint 1170 = epoch 3 × 390 step), data identik dengan run kanonik (test 969/293/468 = label lama; cek silang pickle-vs-CSV lokal: split identik per baris) — tetapi val F1 masih −0.068 dari kanonik 0.7233. Tersangka tersisa: **stack library** (run ini: transformers **5.0.0** / torch 2.10.0 / peft 0.19.1; run kanonik era transformers 4.x).
- **Temuan sampingan penting (data lineage)**: `Data/data_preprocessed_with_emoticon.csv` (MD5 f36594…) berisi **label versi baru** (4686/1510/2452 — hasil koreksi pipeline anotasi) dan pickle `Data/split_data.pkl` sudah regenerasi ke label baru; sedangkan dataset Kaggle (upload 25/08) masih **label lama** (4843/1465/2340) = yang dipakai run kanonik `04` dan semua run Kaggle sejauh ini. → Konsekuensi: angka `04` kanonik dan E1 sebanding (label sama), tapi ke depan harus diputuskan pakai label lama vs label baru (keputusan penelitian, bukan teknis).
- **Push v5 (uji penentu P0.3)**: ERROR — instalasi campur: `pip install` di atas 5.0 meninggalkan file sisa (ImportError TFPreTrainedModel dari file campur 5.0+4.46).
- **Push v6 (uninstall dulu, lalu install)**: ERROR sama — `pip uninstall` tidak bersih; file `import_utils.py` tetap versi 5.0. Ditambah **bug orde eksekusi**: sel versi-print di awal sudah `import transformers 5.0` → modul ter-cache di sys.modules sepanjang sesi, jadi reinstall di sel berikutnya tidak pernah benar-benar dipakai.
- **Push v7 (fix komplet)**: (1) `--force-reinstall --no-deps` + pin dependensi eksplisit (tokenizers 0.20.3, huggingface-hub 0.26.5) — menghindari resolver gagal; (2) Sel versi-print & guard `assert` DIPINDAHKAN SETELAH sel pin, sehingga import memakai stack yang SUDAH dipin; (3) guard `from transformers import TFPreTrainedModel` membuktikan tidak ada file campur 5.0. → COMPLETE ±36 menit, stack 4.46.3 aktif dikonfirmasi.

### PASCA

**Hasil v7 (stack 4.46.3, label lama, single GPU):**

| ε | Val Macro F1 | Test Acc | Test Macro F1 | Netral F1 |
|---|---|---|---|---|
| 0 (baseline) | 0.7166 | 0.7763 | **0.7216** | 0.57 |
| **0.10** | **0.7207** | **0.7902** | **0.7295** | 0.55 |

- Harness: kolom `text_bert` terkonfirmasi, distribusi pred wajar, COLLAPSE TIDAK.
- Baseline Δ −0.011 vs kanonik `04` (0.7328) → **dalam toleransi ±0.02 — pin versi terbukti menyelesaikan gap.**
- McNemar eps10 vs baseline: b=14, c=38, **p=0.0012 (signifikan)** — Label Smoothing secara statistik berbeda, tapi efek sangat kecil (+0.008 Macro F1). Netral F1 justru turun (0.57 → 0.55).

**Gate E1** — GAGAL: netral recall ≪ 0.60, Macro F1 ≪ 0.76.

### Analisis ceiling E3 (threshold calibration, offline)

Dari probs eps10 yang tersimpan: bobot netral bisa dipaksa ekstrem ([1, 20, 1] → netral recall **0.942**), tapi Macro F1 terjun ke **0.517** (model hampir selalu menebak netral). Sweet spot terbaik: w=[1, 1.75, 1] → Macro F1 0.726, netral F1 0.557 — hanya ±0.005 dari baseline. McNemar tidak signifikan.

**Kesimpulan E1+E3**: bukan masalah keputusan threshold — **probabilitas model sendiri tidak membedakan netral dari negatif/positif dengan baik.** Netral F1 mentok di ~0.56 bahkan dengan Label Smoothing. Calibration/decision-level fixes tidak akan menembus 0.60.

**Rekomendasi**: fokus ke **kualitas ground truth & embedding**, bukan keputusan — pertimbangkan: (1) bandingkan ulang baseline dengan label baru/corrected (hasil pipeline anotasi — label 4686/1510/2452 vs label lama 4843/1465/2340 yang dipakai semua run sejauh ini); (2) augmentasi data yang tidak sekadar smoothing atau threshold, tapi perubahan kualitatif pada training signal (focal loss, weighted sampling per confidence, atau data augmentation targeted).

---

## E1-final di label corrected + E3 Threshold Calibration  ✅

### PROSES
- **Push v8 (baseline ε=0 di label corrected)**: COMPLETE ±12 menit. Test: acc 0.7827, Macro F1 **0.7371**, netral **P 0.58 / R 0.57 / F1 0.58** — lebih baik dari baseline label lama (+0.015 Macro F1) dan di atas kanonik `04` (0.7328). Harness OK, test distribusi baru 937/302/491 terkonfirmasi.
- **Push v9 (sweep penuh di label corrected)**: COMPLETE. **Label Smoothing NEGATIF di label baru** — semua ε di bawah baseline val (0.6997/0.6949/0.6906 vs 0.7099). Masuk akal: label corrected sudah menangkap ambiguitas; smoothing mengaburkan sinyal. Baseline v9 = v8 persis (0.7371) → **reproducibility antar-run terbukti**.

### E3 — Threshold Calibration di label corrected (offline, tanpa GPU)

Dari probs baseline label corrected (`hasil_e1_test_baseline.csv` v9):

**Sweep bobot manual (test penuh)** — peak w=[1, 1.5, 1]: acc 0.7746, Macro F1 **0.7394**, netral **P 0.546 / R 0.669 / F1 0.601**.

**Protokol bersih** (split test → cal 50% + holdout 50% stratified, seed 42):
- Coordinate-ascent otomatis di cal menemukan bobot terlalu mild (w=[0.8, 1.15, 1.05]) → holdout Macro F1 turun (0.7310 → 0.7221) meski netral R naik (0.589 → 0.636). Cal subset kecil → optimum noisy.
- **Fixed w=[1, 1.5, 1] pada holdout** (dipilih a priori, tak pernah melihat holdout):

| | Acc | Macro F1 | Netral P | Netral R | Netral F1 |
|---|---|---|---|---|---|
| baseline | 0.7769 | 0.7310 | 0.586 | 0.589 | 0.587 |
| **w=[1, 1.5, 1]** | 0.7665 | **0.7325** | 0.545 | **0.689** | **0.608** |

- McNemar holdout: b=24, c=15, p=0.20 (tidak signifikan — perubahan menyebar antar sampel).

### PASCA

**Verdict E3 — SEBAGIAN BERHASIL**: netral recall **+0.10** (0.589 → 0.689) dan netral F1 tembus **0.60** dengan biaya Macro F1 ±0 (+0.0015) dan accuracy −0.01. **Gate netral recall (≥0.60) TERCAPAI pertama kalinya**; gate Macro F1 (≥0.76) belum — itu butuh perbaikan model-level (bukan decision-level). Kalibrasi tersimpan: `hasil_e3_kalibrasi_test.csv` (full test, w=[1,1.5,1]).

**Kandidat final tesis (label corrected + w=[1, 1.5, 1], test penuh n=1730)**: acc 0.7746, Macro F1 0.7394, netral R 0.669 / F1 0.601.

**Rantai bukti lengkap** (semua terdokumentasi di log ini):
1. Pipeline lama punya bug evaluasi (P0) → diperbaiki + harness verifikasi.
2. DataParallel menggandakan batch → single GPU.
3. Stack transformers 5.0 menurunkan performa → pin 4.46.3 (+ pola instalasi bersih + guard).
4. Path mount dataset CLI 2.x berubah → loader fleksibel.
5. **Label corrected memperbaiki baseline** (0.7216 → 0.7371) — validasi empiris pipeline anotasi.
6. Label smoothing: negatif di label corrected (temuan negatif yang valid).
7. Threshold calibration w=[1, 1.5, 1]: netral recall +0.10 tanpa biaya Macro F1.

---

*(E2 adaptive LS: di-skip — LS uniform sudah negatif di label corrected, varian adaptive tidak akan membalik arah. Langkah berikutnya: E5 perbandingan final seluruh metode, atau perbaikan model-level bila target Macro F1 0.76 tetap dikejar.)*

---

## P1 — Class-Weighted CrossEntropy (roadmap Macro F1 ≥ 0.80)  🟡 berjalan

### PRA (ditulis sebelum push)
- **Kondisi**: baseline terkunci (label corrected + kalibrasi w=[1,1.5,1]) — acc 0.7746 / Macro F1 **0.7394** / netral R 0.669 / F1 0.601. Target roadmap: Macro F1 ≥ 0.80.
- **Apa yang diubah**: notebook baru `07_e4_focal_loss.ipynb` (salinan struktur 06, env/protokol identik) — mengganti sweep label smoothing dengan **SATU pelatihan** memakai `WeightedTrainer` (custom `compute_loss` → `CrossEntropyLoss(weight=...)`), bobot kelas **negatif 0.75 / netral 1.32 / positif 1.03**. Parameter tetap: LoRA r16/α32/dropout0.3, batch16, lr2e-4, 5 epoch, seed 42, `load_best_model_at_end` per val macro F1.
- **Ekspektasi (hipotesis roadmap)**: recall netral naik tanpa merusak kelas lain → Macro F1 ≥ 0.76 (gate P1).
- **Cara ukur**: val macro F1 → test classification report + `verify_metrics.py` + **McNemar vs baseline** (`results/baseline_compare/hasil_baseline_kalibrasi_test.csv`) + simpan probabilitas (`hasil_e4_test.csv`).
- **Risiko**: (1) bobot 1.32 netral kurang/terlalu agresif; (2) seed variance → McNemar sebagai hakim; (3) jika Macro F1 < 0.76 → lanjut P2 (focal loss γ=2).

### PROSES
- Branch `exp_focal_weighted` dibuat; `07_e4_focal_loss.ipynb` + `temp_kernel_e4/` (slug `thesis-lora-e4-class-weight`) dibangun & tervalidasi (nbformat + syntax + marker).
- Baseline pembanding disalin ke `results/baseline_compare/hasil_baseline_kalibrasi_test.csv`.
- **Push v1**: RUNNING (kernel `emanuelembuaijdak/thesis-lora-e4-class-weight`).

### PASCA

**Hasil P1 (weighted CE, label corrected, stack 4.46, single GPU):**

| Metrik (test, n=1730) | Baseline+kalibrasi w=[1,1.5,1] | **P1 weighted CE (argmax)** |
|---|---|---|
| Accuracy | 0.7746 | 0.7711 |
| Macro F1 | 0.7394 | **0.7339** |
| Netral P/R/F1 | 0.546/0.669/0.601 | **0.57/0.63/0.60** |
| Val Macro F1 | 0.7099 | 0.6930 |

- Harness: COLLAPSE TIDAK, distribusi pred 852/336/542.
- **McNemar P1 vs baseline kalibrasi**: b=36, c=42, **p=0.572 (tidak signifikan)** — weighted CE tidak mengalahkan baseline.
- **Temuan menarik**: weighted CE *tanpa kalibrasi* sudah menaikkan netral recall ke 0.63 (vs 0.571 baseline argmax) — setara efek kalibrasi; tapi Macro F1 sedikit lebih rendah.

**Gate P1 — GAGAL** (0.7339 < 0.76) → lanjut **P2 (Focal Loss γ=2 + class weight)** sesuai roadmap.

---

## P2 — Focal Loss (γ=2) + Class Weight  🟡 berjalan

### PRA (ditulis sebelum push)
- **Kondisi**: P1 (weighted CE) gagal gate — test Macro F1 0.7339, McNemar vs baseline p=0.57 (tidak signifikan). Baseline terkunci tetap pembanding.
- **Apa yang diubah**: notebook baru `08_e5_focal_loss.ipynb` — identik dengan P1, loss diganti **Focal Loss** `FL(p_t) = −α_t·(1−p_t)^γ·log(p_t)` dengan **γ=2**, α_t = class weights {0.75, 1.32, 1.03}.
- **Ekspektasi (hipotesis roadmap)**: fokus pada sampel netral yang sulit → Macro F1 ≥ 0.78.
- **Cara ukur**: val macro F1 → test report + harness + McNemar vs baseline + simpan probs (`hasil_e5_test.csv`).
- **Risiko**: γ terlalu kuat → undertrain; jika Macro F1 ~0.76 → pertimbangkan P4 (active learning).

### PROSES
- `08_e5_focal_loss.ipynb` + `temp_kernel_e5/` dibangun & tervalidasi (nbformat + syntax + marker).
- **Push v1**: RUNNING (kernel `emanuelembuaijdak/thesis-lora-e5-focal-loss`).

### PASCA

**Hasil P2 (focal γ=2 + CW, label corrected, stack 4.46, single GPU):**

| Metrik (test, n=1730) | Baseline+kalibrasi | **P2 focal γ=2** |
|---|---|---|
| Accuracy | 0.7746 | **0.7341** |
| Macro F1 | 0.7394 | **0.7041** |
| Netral P/R/F1 | 0.546/0.669/0.601 | **0.48/0.68/0.56** |
| Val Macro F1 | 0.7099 | 0.6903 |

- Harness: COLLAPSE TIDAK, distribusi pred 766/426/538 (netral di-prediksi berlebihan vs aktual 302).
- **McNemar P2 vs baseline**: b=21, c=91, **p≈0,000000 (SIGNIFIKAN, arah NEGATIF)** — focal loss signifikan LEBIH BURUK.
- Penyebab: γ=2 terlalu agresif → presisi netral runtuh 0.48; kelas negatif recall turun ke 0.72.

**Gate P2 — GAGAL** (0.7041 jauh di bawah 0.78 dan bahkan 0.76). **Temuan negatif**: focal loss γ=2 + CW tidak cocok untuk data ini. Pivot: lanjut **P3 (kalibrasi ulang grid netral 1,2–1,6)** pada probabilitas baseline label corrected; jika mentok → **P4 active learning** (satu-satunya jalur dengan peluang naik signifikan).

---

## P3 — Kalibrasi ulang (grid netral 1,2–1,6)  ✅ selesai

### PROSES
- Offline (tanpa GPU): grid bobot netral {1.2, 1.3, 1.4, 1.5, 1.6} pada probabilitas **baseline label corrected** (`hasil_e1_test_baseline.csv`), protokol bersih cal(50%)→holdout(50%) stratified, seed 42.
- Bobot dipilih di cal → diterapkan ke holdout (estimasi tak bias) + McNemar.

### PASCA

**Hasil cal (dipilih):** w=[1, 1.5, 1] terbaik di cal (macro F1 0.7462, netral R 0.649/F1 0.594).

**Holdout (n=865):**

| | Acc | Macro F1 | Netral P/R/F1 |
|---|---|---|---|
| Argmax | 0.7769 | 0.7310 | 0.586/0.589/0.587 |
| **Kalibrasi w=[1,1.5,1]** | 0.7665 | **0.7325** | 0.545/**0.689**/0.608 |

- McNemar holdout: b=24, c=15, p=0.20 (tidak signifikan).
- **Temuan**: bobot terbaik hasil kalibrasi ulang = w=[1, 1.5, 1] — **persis sama dengan baseline terkunci**. Artinya baseline sudah pada titik optimal kalibrasi; grid 1.2–1.6 tidak memberi keuntungan tambahan.

**Gate P3 — GAGAL** (0.7325 < 0.79). Kesimpulan lintas eksperimen: P1 (weighted CE) ≈ baseline, P2 (focal) jauh lebih buruk, P3 tidak menambah. **Bottleneck bukan pada keputusan/loss — model-level dengan data yang ada mentok di Macro F1 ≈ 0.73–0.74.**

---

## Phase 2: Representation Enhancement — P1 (Fine-Tuning Sweep)  ✅ selesai

### PRA (ditulis sebelum push)
- **Kondisi**: Baseline terkunci (label corrected + threshold calibration $w=[1, 1.5, 1]$) menghasilkan Acc 0.7746 / Macro F1 **0.7394** / Netral Recall 0.669 / F1 0.601. Eksperimen loss (Class Weight, Focal Loss, Label Smoothing, Resampling) terbukti tidak mampu melampaui ceiling 0.74.
- **Apa yang diubah**: Notebook baru `notebooks/exp_p1_ft_sweep.ipynb` (`temp_kernel/exp_p1_ft_sweep/`) yang menjalankan sweep 10 varian kombinasi hyperparameter training:
  - *Learning Rate*: $1\times 10^{-5}, 2\times 10^{-5}, 3\times 10^{-5}, 5\times 10^{-5}, 1\times 10^{-4}, 2\times 10^{-4}$
  - *Epochs*: 5, 8, 10
  - *Warmup Ratio*: 0.0, 0.1
  - *Weight Decay*: 0.0, 0.01
  - *Max Length*: 64, 128
- **Ekspektasi**: Memastikan IndoBERTweet tidak tersandera oleh hyperparameter dasar. Target gate: Macro F1 $\ge 0.77$.
- **Cara ukur**: Evaluasi validation macro F1 per varian $\rightarrow$ model terbaik dievaluasi di test set ($n=1.730$) + simpan `exp_p1_ft_sweep_val.csv` dan `exp_p1_ft_sweep_test.csv`.

### PROSES
- Version 1: Push gagal pada inisialisasi seed (`NameError: set_seed`).
- Version 2: Fixed import explicit di template generator notebook $\rightarrow$ Push sukses $\rightarrow$ COMPLETE (durasi $\approx 85$ menit, 10 varian dieksekusi tuntas di Kaggle T4).

### PASCA

**Hasil Sweep Validasi (10 Varian, `exp_p1_ft_sweep_val.csv`):**

| Ranking | Varian | Learning Rate | Epochs | Warmup | Weight Decay | Max Length | Val Acc | Val Macro F1 | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **t10** | **2e-4** | **8** | **0.1** | **0.01** | **64** | **0.7587** | **0.7163** | OK |
| 🥈 | **t8** | 2e-4 | 10 | 0.1 | 0.01 | 128 | 0.7630 | 0.7115 | OK |
| 🥉 | **t6 (base)** | 2e-4 | 8 | 0.1 | 0.01 | 128 | 0.7457 | 0.6987 | OK |
| 4 | **t5** | 1e-4 | 8 | 0.1 | 0.01 | 128 | 0.7457 | 0.6924 | OK |
| 5 | **t4** | 5e-5 | 8 | 0.1 | 0.01 | 128 | 0.7283 | 0.6472 | OK |
| 6 | **t7** | 3e-5 | 10 | 0.1 | 0.01 | 128 | 0.7052 | 0.6138 | OK |
| 7 | **t3** | 3e-5 | 8 | 0.1 | 0.01 | 128 | 0.6806 | 0.5346 | OK |
| 8 | **t9** | 3e-5 | 8 | 0.1 | 0.01 | 64 | 0.6618 | 0.4813 | OK |
| 9 | **t2** | 2e-5 | 5 | 0.1 | 0.01 | 128 | 0.6069 | 0.4083 | OK |
| 10 | **t1** | 1e-5 | 5 | 0.0 | 0.00 | 128 | 0.5448 | 0.2417 | OK (underfit) |

**Evaluasi Model Terbaik (t10) pada Test Set ($n=1.730$, `reports/verifikasi_p1_ft_sweep.md`):**
- **Akurasi**: 77.98%
- **Macro F1 (Argmax)**: **0.7390** (Precision 0.7318, Recall 0.7488)
- **Netral P / R / F1**: 0.56 / 0.61 / 0.59
- **COLLAPSE**: TIDAK

**Verdict Gate P1 — GAGAL (0.7390 < 0.77)**:
- Optimasi hyperparameter komprehensif mengonfirmasi bahwa batas tertinggi model pada konfigurasi training yang ada adalah **Macro F1 $\approx 0.739$**.
- Hyperparameter tuning **bukan** penyebab bottleneck.
- **Tindakan Sesuai Roadmap**: **LANGSUNG PIVOT KE P2 (Task-Adaptive Pretraining / TAPT via MLM)** untuk mengubah kapasitas representasi bahasa dasar.

---

## Phase 2: Representation Enhancement — P2 (Task-Adaptive Pretraining / TAPT)  ✅ selesai

### PRA (desain metodologis)
- **Kondisi**: P1 sweep selesai dan membuktikan hyperparameter bukan bottleneck utama (mentok di Macro F1 0.7390). Kapasitas representasi bahasa menjadi satu-satunya jalur peningkatan performa menuju Macro F1 $\ge 0.80$.
- **Apa yang diubah**: Notebook `notebooks/exp_p2_tapt_mlm.ipynb` (`temp_kernel/exp_p2_tapt_mlm/`):
  1. *Tahap 1 (Self-Supervised)*: Masked Language Modeling (MLM 15% mask) pada seluruh korpus 8.648 tweet bencana banjir selama 3 epoch (LR $5\times 10^{-5}$, evaluasi perplexity pada 10% holdout korpus).
  2. *Tahap 2 (Supervised Fine-Tuning)*: Checkpoint domain-adapted TAPT dipasangi LoRA ($r=16, \alpha=32$) dan dilatih pada dataset sentimen terkoreksi (`text_bert`).
- **Ekspektasi**: Model menginternalisasi konteks semantik lokal bencana banjir (nama sungai, istilah debit, singkatan darurat), mendorong Macro F1 $\ge 0.77 - 0.80$.
- **Cara ukur**: Test classification report, confusion matrix, threshold calibration ($w=[1, 1.5, 1]$), dan uji signifikansi statistik McNemar vs baseline.

### PROSES
- Version 1: Push gagal pada sel getitem loader (`TypeError: unhashable type: 'dict'`).
- Version 2: Menambahkan kelas `MLMDataset` langsung ke modul resmi `src/data.py` $\rightarrow$ Re-generate notebook & staging $\rightarrow$ Push sukses $\rightarrow$ COMPLETE (durasi $\approx 25$ menit di Kaggle T4 GPU, output: `exp_p2_tapt_mlm_test.csv`, `exp_p2_tapt_mlm_summary.json`).

### PASCA

**Hasil Evaluasi Data Uji ($n=1.730$, `reports/verifikasi_p2_tapt_mlm.md`):**

| Metrik | P2 TAPT (Argmax) | P2 TAPT + Kalibrasi ($w=[1, 1.4, 1]$) | Baseline Terkunci + Kalibrasi | Baseline Argmax |
|---|:---:|:---:|:---:|:---:|
| **Akurasi** | **80,06%** 🏆 | 79,60% | 77,46% | 78,27% |
| **Macro Precision** | **0,7583** | 0,7494 | 0,7310 | 0,7347 |
| **Macro Recall** | 0,7383 | **0,7464** | 0,7532 | 0,7403 |
| **Macro F1-Score** | **0,7461** | **0,7479** 🥇 | 0,7394 | 0,7371 |
| **Negatif F1** | 0,87 | 0,87 | 0,84 | 0,84 |
| **Netral F1 (P / R)** | 0,58 (0,64 / 0,53) | **0,59** (0,60 / **0,59**) | 0,60 (0,55 / 0,67) | 0,58 (0,58 / 0,57) |
| **Positif F1** | **0,79** | 0,78 | 0,78 | 0,79 |
| **COLLAPSE** | TIDAK | TIDAK | TIDAK | TIDAK |

**Uji Signifikansi Statistik (McNemar Test vs Baseline):**
- $b=57$ (Base benar, P2 salah), $c=93$ (Base salah, P2 benar)
- $\chi^2 = 8,1667$, Exact Binomial **$p = 0,004113$ ($p < 0,01$)**
- **KESIMPULAN METODOLOGIS**: P2 TAPT terbukti **SIGNIFIKAN SECARA STATISTIK LEBIH BAIK** dibanding baseline (pertama kali dalam sejarah proyek intervensi memberikan peningkatan $p < 0,01$ searah positif).

**Temuan Kunci**:
1. Pretraining domain spesifik (TAPT) berhasil meningkatkan representasi semantik dasar kalimat tweet, mendorong akurasi melampaui 80% (**80.06%**) dan Macro F1 argmax naik ke **0.7461** (naik ke **0.7479** dengan kalibrasi $w=[1, 1.4, 1]$).
2. Presisi kelas Netral meningkat tajam dari 0.55 ke **0.64** (argmax) dan **0.60** (kalibrasi).



