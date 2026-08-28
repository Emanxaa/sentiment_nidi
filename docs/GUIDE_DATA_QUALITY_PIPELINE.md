# Panduan Penggunaan — Data Quality Pipeline

Panduan operasional menjalankan ulang pipeline kualitas data sesuai
`docs/PRD_DATA_QUALITY_PIPELINE.md`. Status saat ini: **Phase 0-6 selesai**,
`data_banjir_v2.csv` sudah aktif sebagai dataset kanonik (data lama di-backup).

---

## 1. Perintah Cepat per Fase

Jalankan dari root proyek (`D:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT`):

| Perintah | Fungsi |
|---|---|
| `python run_all.py --phase 0` | Audit duplicate, missing, noise → `reports/` |
| `python run_all.py --phase 1` | Audit preprocessing LSTM/IndoBERT/stopword → `reports/` |
| `python run_all.py --phase 2` | Sampling 1000 stratified → `Annotation/gold_dataset_1000.csv` |
| `python run_all.py --phase 3` | Anotasi LLM 20 batch (provider dari `.env`) |
| `python run_all.py --phase 3 --dry-run` | Uji alur tanpa API key (simulasi) |
| `python run_all.py --phase 3 --force` | Timpa batch yang sudah ada |
| `python run_all.py --phase 3 --provider openai` | Paksa pakai OpenAI (default: gemini) |
| `python run_all.py --phase 3 --only-batch 5` | Proses hanya batch 5 |
| `python run_all.py --phase 4` | QA + bentuk `data_banjir_v2.csv` |
| `python run_all.py --phase 5` | Agreement, Kappa, label flip → `Annotation/reports/` |
| `python run_all.py --phase 6` | Bangun `split_data_v2.pkl` (tanpa mengubah data kanonik) |
| `python run_all.py --phase 6 --activate-v2` | Backup data lama, aktifkan v2 sebagai kanonik |
| `python run_all.py --phase all6` | Jalankan Phase 0-6 berurutan |

**Dependensi:** `python -m pip install -r requirements.txt` (termasuk `openai`, `google-genai`).

---

## 2. Konfigurasi API Key

Buat file `.env` di root proyek (jangan dikomit — sudah di-gitignore):

```
# Google Gemini (default provider)
GEMINI_API_KEY=AIza...

# Alternatif: OpenAI
# OPENAI_API_KEY=sk-xxxx
```

- Key Gemini gratis: https://aistudio.google.com/apikey
- Provider default: `gemini` (model `gemini-2.5-flash`), bisa diganti `--model`.
- Tanpa key → Phase 3 akan berhenti dengan pesan jelas; `--dry-run` tetap bisa dipakai.

---

## 3. Alur Kerja dari Nol (jika dataset berubah)

1. Taruh data baru sebagai `Data/data_banjir.csv` (schema: `text, clean_text,
   created_at, keyword, processed_text, sentimen, label, emoticon`).
2. Jalankan ulang `01_preprocessing.ipynb` → regenerasi
   `data_preprocessed_with_emoticon.csv`.
3. `python run_all.py --phase 0` dan `--phase 1` → audit & dokumentasi.
4. `python run_all.py --phase 2` → gold dataset 1000.
5. `python run_all.py --phase 3` (isi `.env` dulu) → anotasi LLM.
6. `python run_all.py --phase 4` → buka `Annotation/human_review.csv` di Excel,
   isi kolom `review_label` (`negatif`/`netral`/`positif`), jalankan Phase 4 lagi.
7. `python run_all.py --phase 5` → evaluasi kualitas label.
8. `python run_all.py --phase 6 --activate-v2` → aktifkan v2.

> Catatan Excel: file `human_review.csv` hasil simpan Excel biasanya ber-delimiter
> `;` dan encoding cp1252 — loader pipeline sudah otomatis mendeteksinya.

---

## 4. Retraining Model di Kaggle/Colab (butuh GPU)

### 4.1 Siapkan kernel
1. Buat kernel baru di Kaggle (GPU T4/P100) atau Colab.
2. Upload 3 file ini ke working directory kernel:
   - `Data/split_data_v2.pkl`
   - `Data/data_preprocessed_with_emoticon_v2.csv`
   - Notebook `02_model_lstm.ipynb`, `03_model_bilstm.ipynb`, `04_model_indobertweet_lora.ipynb`
3. Karena v2 sudah diaktifkan sebagai kanonik, nama file di notebook sudah cocok
   (`split_data.pkl`, `data_preprocessed_with_emoticon.csv`). Jika Anda meng-upload
   dengan nama v2, rename dulu:
   ```
   !mv split_data_v2.pkl split_data.pkl
   !mv data_preprocessed_with_emoticon_v2.csv data_preprocessed_with_emoticon.csv
   ```

### 4.2 Jalankan & catat hasil
4. Jalankan notebook secara berurutan (LSTM → BiLSTM → IndoBERTweet-LoRA).
5. Ambil **Macro F1** dari output masing-masing notebook (laporan klasifikasi / ringkasan).

### 4.3 Tabel perbandingan (isi untuk tesis)

| Model | Macro F1 baseline (v1) | Macro F1 v2 | Delta |
|---|---|---|---|
| LSTM | ... | ... | ... |
| BiLSTM | ... | ... | ... |
| IndoBERTweet-LoRA | ... | ... | ... |

> Baseline v1 bisa didapat dengan menjalankan notebook yang sama memakai file
> backup: `Data/split_data_v1_backup.pkl` → `split_data.pkl`.

---

## 5. Kembali ke Data Lama (revert)

v2 aktif = `data_banjir.csv`, `data_preprocessed_with_emoticon.csv`,
`split_data.pkl` sudah berisi data v2. Untuk kembali ke v1:

```
copy Data\data_banjir_v1_backup.csv Data\data_banjir.csv
copy Data\data_preprocessed_with_emoticon_v1_backup.csv Data\data_preprocessed_with_emoticon.csv
copy Data\split_data_v1_backup.pkl Data\split_data.pkl
```

---

## 6. Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `429 insufficient_quota` (OpenAI) | Akun kehabisan kuota/billing → pakai Gemini (`.env` → `GEMINI_API_KEY`) |
| `GEMINI_API_KEY belum diset` | Key belum ada di `.env` |
| `UnicodeDecodeError` / parser error saat baca CSV hasil Excel | File disimpan Excel (delimiter `;`/cp1252) — loader sudah otomatis mendeteksi; kalau masih gagal, simpan ulang sebagai "CSV UTF-8" |
| `OSError Errno 22` saat menulis file | File sedang dikunci (terbuka di Excel/OneDrive) → tutup lalu ulangi |
| Phase 6 lama | Stemming Sastrawi; pipeline sudah paralel (stem sekali per token unik) — wajar beberapa menit |
| Batch dilewati saat menjalankan Phase 3 | Resume aktif; gunakan `--force` untuk menimpa |

---

## 7. Lokasi Output (untuk rujukan tesis)

| Deliverable | Path |
|---|---|
| Laporan audit data | `reports/data_quality_audit.md` (+ duplicate/missing/noise CSV) |
| Laporan audit preprocessing | `reports/preprocessing_audit.md` (+ `critical_stopword_report.csv`) |
| Gold dataset | `Annotation/gold_dataset_1000.csv` |
| Anotasi LLM (20 batch) | `Annotation/results/batch_001.csv ... batch_020.csv` |
| Prompt & respons mentah | `Annotation/prompts/`, `Annotation/batches/` |
| QA & evaluasi | `Annotation/reports/qa_report.md`, `evaluation_report.md`, `label_flip_analysis.csv` |
| Dataset v2 | `Data/data_banjir_v2.csv` |
| Artefak retraining | `Data/split_data_v2.pkl`, `data_preprocessed_with_emoticon_v2.csv` |
| Backup v1 | `Data/*_v1_backup.*` |
