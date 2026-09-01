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

*(entri E2 dst ditambahkan setelah E1 dievaluasi)*
