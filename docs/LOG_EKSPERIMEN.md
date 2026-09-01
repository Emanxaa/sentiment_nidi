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
- **Push v2**: RUNNING ulang (melewati titik gagal v1).

### PASCA
- *(menunggu)*

---

*(entri E2 dst ditambahkan setelah E1 dievaluasi)*
