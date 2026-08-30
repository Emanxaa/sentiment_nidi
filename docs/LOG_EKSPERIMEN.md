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
- v8 di-push dengan `machine_shape: NvidiaTeslaT4` + internet on → RUNNING, dipantau otomatis.

### PASCA
- *(menunggu run selesai)*

---

## E1 — Label Smoothing (ε ∈ {0.05, 0.10, 0.15} + baseline)  🟡 pra-eksperimen

### PRA (ditulis sebelum push)
- **Kondisi**: E0 = baseline kanonik run v8 (angka final diisi pasca verifikasi Fase V). Konfigurasi terbaik tuning: trial-4 (batch 16, dropout 0.3, lr 2e-4, r 16, α 32). Bottleneck: netral recall ~0.50.
- **Apa yang diubah**: notebook baru `06_e1_label_smoothing.ipynb` — identik dengan pipeline empiris v1 (data `text_bert`, split 80:20 + 10% val, seed 42, `load_best_model_at_end`) tetapi: (a) tanpa grid search/simulasi/class-weight; (b) 4 pelatihan dengan `label_smoothing_factor` ε = {0.0, 0.05, 0.10, 0.15}; (c) model terbaik-ε vs baseline dievaluasi di test + simpan prediksi & probabilitas (`hasil_e1_test_*.csv`); (d) McNemar exact inline.
- **Ekspektasi**: Macro F1 +1–3 poin; recall netral naik; precision negatif sedikit turun (efek smoothing yang diketahui). Target gate: recall netral ≥ 0.60 & Macro F1 ≥ 0.76.
- **Cara ukur**: val macro F1 (pemilihan ε) → test classification report + `verify_metrics.py` pada CSV hasil pull + McNemar vs baseline.
- **Risiko**: (1) smoothing merusak kelas mayoritas → deteksi via per-kelas metrik; (2) perbedaan seed antar-run Kaggle → toleransi ±0.02, McNemar sebagai hakim; (3) 4 pelatihan dalam satu run → ±2× waktu baseline.

### PROSES
- *(belum di-push — menunggu E0 terkunci)*

### PASCA
- *(menunggu)*

---

*(entri E2 dst ditambahkan setelah E1 dievaluasi)*
