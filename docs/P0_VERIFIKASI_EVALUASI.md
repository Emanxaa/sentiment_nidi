# P0 — Verifikasi & Perbaikan Pipeline Evaluasi

Dokumen ini menuntaskan inkonsistensi **"Best Validation Macro F1 = 0.2393"** vs confusion matrix **Macro F1 ~0.70**, dan menyediakan alat verifikasi metrik agar keputusan tuning ke depan tidak bias.

---

## 1. Root cause: 0.2393 vs 0.70 (dua model berbeda)

| Angka | Sumber | Arti |
|---|---|---|
| **Macro F1 0.2393** | `results/thesis-lstm-v2.ipynb` (grid search **LSTM**) | Model LSTM **collapse**: selalu menebak kelas mayoritas (negatif). Val accuracy ~0.56 = proporsi negatif, loss nyangkut di ~ln(3) (output uniform). Semua 24 konfigurasi Baseline menghasilkan `0.239308` identik — tanda klasik collapse. Kode metriknya **benar** (`argmax(logits, axis=1)` + `precision_recall_fscore_support(average='macro', zero_division=0)`); model-nya yang tidak pernah belajar (EarlyStopping `patience=3, restore_best_weights=True` mengembalikan bobot ~epoch-1 yang hampir random). |
| **Macro F1 ~0.70** | `results/thesis-indobertweet-lora-v1.ipynb` sel 24–27 (**IndoBERTweet-LoRA**) | Evaluasi test 1730 sampel memakai `trainer_best`. Valid & konsisten dengan best validation LoRA (0.6806 di v1; 0.7233 di notebook utama `04`). |

**Kesimpulan:** tidak ada kontradiksi metrik di pipeline LoRA. 0.2393 adalah hasil grid search LSTM yang collapse — bukan bug perhitungan F1 dan bukan hasil LoRA. Karena semua 120 eksperimen LSTM ≤ 0.2393, "best config" LSTM tidak bermakna sampai training-nya diperbaiki (langkah terpisah, lihat §6).

### Rumus tanda collapse (untuk deteksi manual)
Jika model selalu menebak kelas mayoritas berproporsi `p` pada masalah 3 kelas:

```
acc      = p
macro F1 = (2p / (1 + p)) / 3
```

Dengan `p = 0.56` → macro F1 ≈ **0.239**. Jika nilai F1 sebuah trial berada di sekitar ini, model collapse.

---

## 2. Angka kanonik IndoBERTweet-LoRA (kolom `text_bert`)

Kolom kanonik untuk input BERT: **`text_bert`** (sesuai notebook utama `04_model_indobertweet_lora.ipynb` yang memuat dari `split_data.pkl`).

| Metrik (test, n=1730) | Nilai kanonik (`04`, `text_bert`) | Nilai v1 lama (`clean_text` — superseded) |
|---|---|---|
| Accuracy | **0.795376** | 0.779769 |
| Precision Macro | **0.745009** | 0.729263 |
| Recall Macro | **0.725157** | 0.692936 |
| Macro F1 | **0.732810** | 0.699401 |
| Negatif (P/R/F1) | 0.85 / 0.89 / 0.87 | 0.84 / 0.89 / 0.86 |
| Netral (P/R/F1) | 0.62 / 0.50 / 0.55 | 0.64 / 0.39 / 0.48 |
| Positif (P/R/F1) | 0.76 / 0.78 / 0.77 | 0.71 / 0.81 / 0.75 |

> Angka 0.779769/0.699401 yang sempat dipakai berasal dari run Kaggle v1 yang **diam-diam memakai `clean_text`** (representasi LSTM), karena auto-deteksi kolom BERT tidak menemukan `text_bert` dan silent fallback ke kolom LSTM. Run ini sudah diperbaiki di P0 (lihat §4).

### 2.1 Hasil re-run Kaggle v5 (30/08) — angka terbaru

Sumber: `results/thesis-indobertweet-lora-v1.ipynb` (notebook hasil run COMPLETE, 69 sel, kolom `text_bert`). Test set n=1730.

**Tuning 6 trial (validation macro F1, terbaik → terendah):**

| Trial | batch_size | dropout | lr | r | α | Acc | Prec | Rec | **F1 macro** |
|---|---|---|---|---|---|---|---|---|---|
| **4** | 16 | 0.3 | 2e-4 | 16 | 32 | 0.7211 | 0.6462 | 0.6280 | **0.6326** |
| 2 | 16 | 0.2 | 1e-4 | 8 | 16 | 0.7110 | 0.6265 | 0.5965 | 0.5878 |
| 5 | 32 | 0.2 | 1e-4 | 16 | 32 | 0.7095 | 0.6427 | 0.5714 | 0.5341 |
| 3 | 16 | 0.3 | 1e-4 | 16 | 32 | 0.6922 | 0.5184 | 0.5494 | 0.5036 |
| 1 | 16 | 0.2 | 5e-5 | 8 | 16 | 0.6416 | 0.4139 | 0.5126 | 0.4559 |
| 6 | 32 | 0.3 | 1e-4 | 16 | 32 | 0.6329 | 0.4134 | 0.5194 | 0.4531 |

**Evaluasi test (retrain trial-4, empiris):**

| Metrik (test, n=1730) | Re-run v5 (30/08) | Kanonik `04` | v1 lama (`clean_text`) |
|---|---|---|---|
| Accuracy | **0.725434** | 0.795376 | 0.779769 |
| Precision Macro | **0.651853** | 0.745009 | 0.729263 |
| Recall Macro | **0.633793** | 0.725157 | 0.692936 |
| Macro F1 | **0.636004** | 0.732810 | 0.699401 |
| Negatif (P/R/F1) | 0.8055 / 0.8421 / 0.8234 | 0.85 / 0.89 / 0.87 | 0.84 / 0.89 / 0.86 |
| Netral (P/R/F1) | 0.4948 / 0.3242 / 0.3918 | 0.62 / 0.50 / 0.55 | 0.64 / 0.39 / 0.48 |
| Positif (P/R/F1) | 0.6552 / 0.7350 / 0.6928 | 0.76 / 0.78 / 0.77 | 0.71 / 0.81 / 0.75 |

> ⚠️ **Gap vs kanonik:** Macro F1 v5 = 0.636 vs kanonik `04` = 0.733 → selisih **-0.097**, jauh melewati toleransi V1 (±0.02). Belum lolos gate verifikasi; perlu diagnosa (kemungkinan: beda versi dataset, seed/struktur notebook, atau `text_bert` vs kolom lain). Jangan kunci angka ini sebagai final sebelum Fase V1 tuntas.

**Class weight (test):** Accuracy 0.7145 · Prec macro 0.6363 · Rec macro 0.6329 · **Macro F1 0.6326** — tidak lebih baik dari empiris (0.636).
**Skenario simulasi:** 1:1:1 → F1 0.5908 · 6:3:1 → F1 0.4911 · 8:1:1 → F1 0.4591 — semua kalah dari baseline empiris.

---

## 3. Harness verifikasi: `quality_pipeline/verify_metrics.py`

Skrip mandiri untuk memverifikasi CSV hasil prediksi secara offline. Mendeteksi:
- collapse (Macro F1 ≤ baseline mayoritas),
- kelas yang tidak pernah diprediksi,
- accuracy tidak lebih baik dari tebakan mayoritas,
- label di luar rentang (indikasi label mapping salah).

### Cara pakai

```cmd
cd /d "D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT"

python -m quality_pipeline.verify_metrics ^
  --preds hasil_prediksi_test_indobertweet_lora.csv ^
  --y-true label_aktual ^
  --y-pred label_prediksi ^
  --label-names negatif,netral,positif
```

| Argumen | Wajib | Keterangan |
|---|---|---|
| `--preds` | ya | Path CSV berisi label aktual & prediksi (angka 0/1/2). |
| `--y-true` | ya | Nama kolom label aktual. |
| `--y-pred` | ya | Nama kolom label prediksi. |
| `--label-names` | tidak | Nama kelas, dipisah koma (default `negatif,netral,positif`). |
| `--out` | tidak | Nama laporan di `reports/` (default `verifikasi_evaluasi.md`). |
| `--dry-run` | tidak | Cetak saja, tanpa menulis laporan. |

Output: ringkasan ke terminal (distribusi, accuracy, Macro/Weighted F1, baseline mayoritas, flag COLLAPSE) + laporan lengkap markdown ke `reports/verifikasi_evaluasi.md` (classification report + confusion matrix).

### Contoh hasil

```
Accuracy      : 0.565700  (baseline 0.565700)
Macro F1      : 0.240872  (baseline 0.240872)
COLLAPSE      : YA
  ! COLLAPSE: Macro F1 0.2409 <= baseline mayoritas 0.2409 - model hampir pasti selalu menebak satu kelas.
```

---

## 4. Perbaikan yang sudah dilakukan di P0

### `results/thesis-indobertweet-lora-v1.ipynb` (notebook Kaggle)
| Sel | Masalah | Perbaikan |
|---|---|---|
| 2 | Auto-deteksi kolom BERT silent fallback ke `clean_text` (kolom `text_bert` tidak pernah dipilih). | `col_bert` kini **eksplisit**: `"text_bert"`; jika tidak ada, `raise ValueError` — tidak ada fallback diam-diam. |
| 26 | `y_test.value_counts()` crash (y_test numpy). | Dibungkus `pd.Series(...)`. |
| 29 | Evaluasi memakai `trainer` (sisa trial terakhir tuning = trial 6, konfigurasi terburuk) → macro F1 0.4587, netral F1 0.0, prediksi salah disimpan ke CSV. | Ganti ke `trainer_best.predict(test_dataset)` + tambah sanity check distribusi prediksi vs aktual. Hasil sekarang konsisten dengan sel 24–27 (~0.70). |
| 33 | `X_train_bert.reset_index(drop=True)` crash (numpy). | `pd.Series(...)`. |
| 48 | `X_test_bert.reset_index(drop=True)` / `y_test.reset_index(drop=True)` crash (numpy). | `pd.Series(...)`. |

### Grid search LSTM (`02_model_lstm.ipynb`, `results/thesis-lstm-v2.ipynb`, `temp_kernel_lstm/02_model_lstm_kaggle.ipynb`)
- Sel loop training: ditambahkan sanity check per trial — distribusi prediksi val + baseline mayoritas + status `COLLAPSE (F1 <= baseline)`.
- Sel ringkasan best: ditambahkan catatan agar best F1 dibandingkan dengan baseline mayoritas sebelum dipakai.

### Tidak diubah
- `04_model_indobertweet_lora.ipynb` — sudah benar, menjadi referensi kanonik.
- Catatan: sel-58 notebook `04` berisi tabel placeholder LSTM hardcoded (angka tidak nyata) — jangan dipakai sebagai angka final; akan diisi setelah training LSTM diperbaiki.

---

## 5. Langkah re-run Kaggle (setelah fix)

1. Push ulang notebook yang sudah diperbaiki (lihat `docs/GUIDE_KAGGLE_CLI.md`):
   ```cmd
   kaggle kernels push -p <folder_kernel>
   ```
2. Pastikan dataset berisi `data_preprocessed_with_emoticon.csv` yang memiliki kolom `text_bert` (sudah ada di `kaggle_dataset/`).
3. Setelah run selesai: cek sel-2 mencetak `Kolom terpilih -> ... BERT: 'text_bert'`, sel-24/27 memberi angka ≈ 0.7954 / 0.7328, dan sel-29 konsisten dengan sel-24.
4. Unduh `hasil_prediksi_test_indobertweet_lora.csv` dan verifikasi offline:
   ```cmd
   python -m quality_pipeline.verify_metrics --preds hasil_prediksi_test_indobertweet_lora.csv --y-true label_aktual --y-pred label_prediksi
   ```
   Harus menunjukkan `COLLAPSE : TIDAK` dan Macro F1 ≈ 0.73.
5. Update tabel angka final di bagian §2 dokumen ini bila hasil re-run berbeda.

---

## 6. Troubleshooting

| Gejala | Penyebab | Tindakan |
|---|---|---|
| Macro F1 ≈ 0.239 dan accuracy ≈ 0.56 pada model 3 kelas | Model collapse (selalu menebak kelas mayoritas). | Cek distribusi prediksi & training loss; tambah epochs, turunkan LR, cek EarlyStopping (`patience`/`restore_best_weights`); sanity check sudah otomatis tercetak di grid search LSTM. |
| Macro F1 0.4587, netral F1 0.0, "netral tak pernah diprediksi" | Evaluasi memakai `trainer` sisa loop tuning, bukan `trainer_best`. | Pastikan evaluasi final memakai `trainer_best` (sudah diperbaiki di sel-29 v1). |
| Angka evaluasi beda antara notebook utama dan versi Kaggle | Kolom input BERT berbeda (`text_bert` vs `clean_text`). | Pakai `text_bert` di kedua versi; cek cetakan `Kolom BERT terpilih`. |
| CSV prediksi tidak bisa diverifikasi harness | Kolom tidak ada / label bukan angka 0/1/2. | Cek pesan error harness (mencantumkan kolom tersedia); pastikan label int. |
| Best config tuning tidak masuk akal | Semua trial collapse sehingga "best" hanya yang paling tidak buruk. | Jangan pakai hasil tuning model collapse; perbaiki training dulu, lalu ulangi tuning. |

---

## 7. Checklist P0

- [x] Cek `average` pada F1 — sudah benar (`average='macro'`, `zero_division=0`) di semua pipeline.
- [x] Cek `label2id` konsisten — mapping `0=negatif, 1=netral, 2=positif` sama di training/validasi/test.
- [x] Cek validasi & test tidak tertukar — val = stratified 10% dari train, test = hold-out 20% (1730 sampel).
- [x] Simpan prediction & label — `compute_metrics` HF menghitung dari logits+labels tiap epoch; CSV prediksi test kini disimpan dari `trainer_best` (bukan trainer salah).
- [x] Hitung ulang F1 dari prediksi mentah — `verify_metrics.py` menghitung ulang dari CSV.
- [x] Deteksi collapse — sanity check inline di grid search LSTM + flag di harness.
- [ ] Re-run `results/thesis-indobertweet-lora-v1.ipynb` di Kaggle → verifikasi angka kanonik (`text_bert`). *(30/08: re-run v5 sudah COMPLETE — hasil 0.636 belum lolos gate ±0.02 vs kanonik 0.733, diagnosa berjalan)*

---

## 8. Langkah berikutnya (di luar P0)

1. **Perbaikan training LSTM/BiLSTM** yang collapse (grid search v2 + percobaan Optuna `0.239x`) — sebelum memakai LSTM sebagai baseline tesis.
2. **Enhancement E1–E5**: Label Smoothing (E1), Adaptive Label Smoothing (E2), Threshold Calibration (E3), Label Confusion Modeling (E4), kombinasi terbaik (E5) — dengan metrik Macro F1, Weighted F1, recall netral, ECE, dan McNemar Test.
