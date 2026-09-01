# Plan Pasca Re-run Kaggle — thesis-indobertweet-lora-v1

**Status (30/08):** re-run v5 sudah **COMPLETE**; hasil ditarik dan tercatat di `docs/P0_VERIFIKASI_EVALUASI.md` §2.1 (Macro F1 test **0.636**, selisih -0.097 vs kanonik 0.733 → **belum lolos gate V1**, diagnosa dulu sebelum lanjut E1–E3).

Rencana kerja setelah run `emanuelembuaijdak/thesis-indobertweet-lora-v1` (notebook LoRA yang sudah diperbaiki P0) selesai di Kaggle. Prinsip: **angka baru dipakai untuk keputusan setelah lolos verifikasi** (Fase V1).

---

## 1. Yang kamu jalankan saat status COMPLETE

```cmd
cd /d "D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT"
kaggle kernels status emanuelembuaijdak/thesis-indobertweet-lora-v1
kaggle kernels output emanuelembuaijdak/thesis-indobertweet-lora-v1 -p .kaggle-outputs\
```

File yang mendarat di `.kaggle-outputs\` (dan yang akan dibaca):

| File | Isi | Kegunaan |
|---|---|---|
| `thesis-indobertweet-lora-v1.ipynb` | Notebook ter-eksekusi dengan **seluruh output sel** | Verifikasi utama: konfirmasi kolom, tabel tuning, report, sanity check |
| `thesis-indobertweet-lora-v1.log` | Log run Kaggle | Cek crash, GPU, urutan eksekusi |
| `hasil_prediksi_test_indobertweet_lora.csv` | Prediksi per-sampel (dari sel-29) | Verifikasi harness + McNemar nanti |
| `hasil_tuning_indobertweet_lora_empiris.csv` | Tabel 6 trial tuning | Cek best trial & val F1 |
| `hasil_indobertweet_lora_empiris_with_emoticon.csv` | Ringkasan metrik empiris | Angka final |
| `hasil_simulasi_indobertweet_lora.csv`, `prediksi_simulasi_indobertweet_lora.csv` | Hasil skenario 1:1:1 / 6:3:1 / 8:1:1 | Ablation balancing |
| `hasil_indobertweet_lora_class_weight.csv`, `perbandingan_class_weight.csv` | Hasil class weight | Ablation balancing |

Setelah tarik, cukup bilang "hasil sudah ditarik" — sisanya dibaca dan dikerjakan dari situ.

---

## 2. Fase V1 — Verifikasi hasil run (gate: harus lulus semua)

1. **Kolom kanonik** — sel-2 mencetak `BERT: 'text_bert'` (bukan fallback `clean_text`).
2. **Log bersih** — tidak ada exception di bagian empiris; sanity check tiap sel jalan.
3. **Tuning sehat** — tabel 6 trial: best val macro F1 di kisaran 0.68–0.72 dan **semua trial jauh di atas baseline mayoritas 0.239** (tidak ada trial collapse). *(Aktual v5: best 0.6326, semua trial sehat — tapi di bawah kisaran 0.68–0.72)*.
4. **Evaluasi final konsisten** — sel-24 (classification report) dan sel-27 (ringkasan) identik dengan sel-29 (bukan 0.4587 lagi); distribusi prediksi tercetak.
5. **Rekonstruksi harness**:
   ```cmd
   python -m quality_pipeline.verify_metrics --preds ".kaggle-outputs\hasil_prediksi_test_indobertweet_lora.csv" --y-true label_aktual --y-pred label_prediksi
   ```
   Harus: `COLLAPSE : TIDAK`, Macro F1 hasil harness == angka report (selisih ≤ 0.001).
6. **Perbandingan tiga angka**:
   - Kanonik `04` (`text_bert`, Colab): acc 0.795376 / Macro F1 0.732810
   - v1 lama (`clean_text`, superseded): acc 0.779769 / Macro F1 0.699401
   - Re-run v5 (30/08): acc 0.725434 / Macro F1 0.636004 — **selisih -0.097 vs kanonik, TIDAK lolos toleransi ±0.02** → diagnosa (bagian 5), jangan lanjut.

**Gate V1:** lulus semua → angka final **dikunci**, `P0_VERIFIKASI_EVALUASI.md` §2 diupdate, lanjut Fase V2. Gagal → berhenti, diagnosa.

---

## 3. Fase V2 — Analisis pasca-verifikasi

- Delta per-kelas (negatif/netral/positif P/R/F1) antara run baru vs kanonik → konfirmasi bottleneck tetap di netral.
- Distribusi prediksi (bergeser atau tidak dari pola 1018/177/535) → bahan kalibrasi.
- Tabel kontingensi per-sampel dari CSV prediksi → disiapkan untuk uji McNemar antar-model (E1–E3 nanti).
- **Catatan teknis:** CSV prediksi saat ini hanya menyimpan label (tanpa probabilitas), jadi ECE/threshold belum bisa dihitung dari run ini. Penyimpanan `softmax(logits)` akan ditambahkan di versi notebook E1.

---

## 4. Fase E1–E3 — Enhancement (run Kaggle berikutnya)

| Eksperimen | Perubahan | Kriteria lanjut |
|---|---|---|
| **E1 Label Smoothing** | `label_smoothing_factor` pada `TrainingArguments`, ε ∈ {0.05, 0.10, 0.15}, konfigurasi trial-4; + sel simpan probabilitas | Macro F1 ≥ 0.76 dan netral recall ≥ 0.60 |
| **E2 Adaptive LS** | Hanya jika netral masih lemah setelah E1 | Netral recall ≥ 0.60 |
| **E3 Threshold Calibration** | Optimasi threshold per kelas di **validation** (bukan test), terapkan ke test; uji McNemar vs default | Recall netral naik tanpa Macro F1 turun > 0.01 |

Target keseluruhan: minimal acc ≥ 0.80, netral recall ≥ 0.60, Macro F1 ≥ 0.76; ideal Macro F1 ≥ 0.80.

**Kapan imbalance masuk:** hanya jika netral recall < 0.55 setelah E1–E3 — dan berupa **class-weight ablation** (kode sudah ada di notebook `04`), bukan SMOTE/oversampling. Bukti eksperimen yang sudah ada: semua perlakuan balancing kalah dari baseline (0.733), jadi ini prioritas terakhir.

Catatan: perbaikan LSTM yang collapse tetap follow-up terpisah, tidak menghalangi E1–E3.

---

## 5. Troubleshooting run

| Gejala | Kemungkinan | Tindakan |
|---|---|---|
| Metrik jauh di bawah kanonik (Δ > 0.02) + `train_batch_size: 32` di `trainer_state.json` (batch 16 × 2 GPU) | **DataParallel T4 x2**: batch efektif digandakan → update optimizer separuh → undertrained | Paksa 1 GPU: sel `CUDA_VISIBLE_DEVICES=0` sebagai sel kode pertama (sudah diterapkan di semua notebook) |
| `'DataParallel' object has no attribute 'config'` di custom Trainer | Trainer membungkus model dengan DP (2 GPU) | Sama: paksa 1 GPU, atau akses via `model.module.config` |
| `ImportError: ... torchao ... only versions above 0.16.0 are supported` (mati ±47 detik, saat build model) | torchao 0.10.0 di image Kaggle tidak kompatibel dengan peft | Pastikan sel `!pip uninstall -y torchao` **aktif** (uncommented) sebelum sel build model — sudah difix; push ulang |
| `FileNotFoundError: /kaggle/input/<slug>/...` (mati <2 menit, dataset status `ready`, metadata benar) | **CLI 2.x memakai path mount baru**: `/kaggle/input/datasets/<owner>/<slug>/` (bukan `/kaggle/input/<slug>/` skema lama) | Jangan hardcode `DATASET_DIR` — cari file dari `os.walk("/kaggle/input")` (fix P0.2 di notebook E1) |
| Status `error`, log berhenti di awal | Dataset tidak attach / CSV berubah | Cek `dataset_sources` di kernel-metadata.json & kolom CSV di log |
| Angka jauh dari kanonik (>0.02) | Versi dataset / seed / kolom beda | Bandingkan distribusi train/test di log vs `04`; cek cetakan kolom sel-2 |
| Report parsial tapi notebook selesai | Sebagian seksi crash (simulasi/class-weight) | Baca log lokasi crash; bagian empiris tetap dipakai kalau lulus cek 1–6 |
| `hasil_prediksi_test_...csv` tidak ada | Sel-29 tidak tereksekusi | Cek log; sel-29 bisa dijalankan ulang terpisah |

---

## 6. Lokasi output

- `reports/verifikasi_evaluasi.md` — laporan harness (dibuat/di-update saat Fase V1).
- `docs/P0_VERIFIKASI_EVALUASI.md` §2 — angka final yang dikunci.
- `.kaggle-outputs/` — artefak mentah hasil pull (jangan diedit).
