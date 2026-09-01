# TASK BOARD — Roadmap & Backlog Repositori Thesis-LSTM-IndoBERT

Dokumen ini memetakan backlog teknis terstruktur untuk pemeliharaan, peningkatan kualitas arsitektur, dan pemenuhan standar reproduksibilitas tesis.

---

## P0 Critical (Reproduksibilitas, Integritas Metrik & Tesis)

### P0-1: Eksekusi & Kunci Ulang Benchmark Baseline LSTM & BiLSTM v2
- **Goal**: Menjalankan ulang pelatihan model baseline LSTM dan BiLSTM menggunakan konfigurasi perbaikan (EarlyStopping `patience=7`, `reduce_lr=True`, data *label corrected*) untuk mengeliminasi riwayat *majority collapse* (Macro F1 0.2393) dan mengunci angka perbandingan yang adil di Bab IV tesis.
- **Files**:
  - [`configs/exp_lstm_v2.yaml`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/configs/exp_lstm_v2.yaml)
  - [`configs/exp_bilstm_v2.yaml`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/configs/exp_bilstm_v2.yaml)
  - [`tools/generate_notebook.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools/generate_notebook.py)
  - [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LAPORAN_AKHIR_EKSPERIMEN.md)
  - [`docs/LOG_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LOG_EKSPERIMEN.md)
- **Risk**: Training model sekuensial pada representasi teks Keras memerlukan waktu dan tuning hyperparameter yang tepat agar konvergen tanpa overfitting.
- **Success metric**: Macro F1 test LSTM & BiLSTM > 0.2393 (bebas collapse), laporan harness `verify_metrics.py` berstatus `OK`, dan tabel perbandingan seluruh model di [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/LAPORAN_AKHIR_EKSPERIMEN.md) terisi lengkap.

---

### P0-2: Strict Column Schema Guard (`text_bert` vs `clean_text`)
- **Goal**: Memasang validasi skema kolom input yang ketat pada loader PyTorch dan TensorFlow/Keras untuk mencegah insiden *silent fallback* ke kolom representasi yang salah (misal: model Transformer memuat `clean_text` alih-alih `text_bert`).
- **Files**:
  - [`src/data.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src/data.py)
  - [`src/keras_data.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src/keras_data.py)
  - [`tools/generate_notebook.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools/generate_notebook.py)
- **Risk**: Script atau eksperimen lama yang mengandalkan fleksibilitas auto-detect kolom akan gagal mengeksekusi dan harus diperbarui secara eksplisit.
- **Success metric**: Loader melempar exception keras (`KeyError` / `ValueError`) jika kolom kanonik tidak ditemukan; 0% kemungkinan *silent fallback* kolom saat runtime.

---

### P0-3: Pemutakhiran Root [`README.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/README.md)
- **Goal**: Memperbarui file `README.md` utama di root repositori agar memuat ikhtisar riset lengkap, spesifikasi arsitektur, panduan instalasi/quickstart, hasil akhir model terbaik, serta indeks navigasi ke seluruh dokumen teknis di [`docs/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs).
- **Files**:
  - [`README.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/README.md)
  - [`docs/README_SUMMARY.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/README_SUMMARY.md)
  - [`docs/HANDOVER.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/HANDOVER.md)
- **Risk**: Duplikasi narasi jika tidak disusun dengan prinsip *gateway/index-first*.
- **Success metric**: Root `README.md` menyajikan dokumentasi yang komprehensif, instruksi CLI yang dapat direproduksi secara mandiri, dan tautan aktif ke semua dokumen referensi.

---

## P1 Important (Kerapian Arsitektur, Refaktorisasi & Kualitas Kode)

### P1-1: Konsolidasi Direktori Staging Kernel Kaggle
- **Goal**: Menghapus direktori staging lama yang terpecah (`temp_kernel_e1/`, `temp_kernel_e4/`, `temp_kernel_e5/`, `temp_kernel_lora/`, `temp_kernel_lstm/`, `temp_kernel_smote/`) dan menstandarisasi seluruh paket kernel di bawah direktori terpadu [`temp_kernel/<exp_id>/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/temp_kernel).
- **Files**:
  - [`temp_kernel/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/temp_kernel)
  - [`tools/generate_notebook.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools/generate_notebook.py)
  - [`docs/GUIDE_KAGGLE_CLI.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/GUIDE_KAGGLE_CLI.md)
- **Risk**: Script lokal manual yang masih menunjuk path lama akan error jika path tidak disesuaikan.
- **Success metric**: Struktur repositori bebas dari folder `temp_kernel_*` legacy; semua kernel dikelola secara eksklusif oleh `tools/generate_notebook.py` di `temp_kernel/<exp_id>/`.

---

### P1-2: Penyatuan Kamus & Modul Preprocessing Teks
- **Goal**: Menggabungkan kamus normalisasi, ekspansi emoticon, dan stopwords yang terpisah antara [`Preprocessing/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Preprocessing) dan [`quality_pipeline/preprocess.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/quality_pipeline/preprocess.py) ke dalam satu paket terpadu `src/preprocessing/`.
- **Files**:
  - [`Preprocessing/emoji_dict.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Preprocessing/emoji_dict.py)
  - [`Preprocessing/normalisasi_dict.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Preprocessing/normalisasi_dict.py)
  - [`quality_pipeline/preprocess.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/quality_pipeline/preprocess.py)
  - [`src/data.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src/data.py)
- **Risk**: Perubahan pemetaan kamus berpotensi sedikit mengubah output tokenisasi pada teks baru jika tidak diuji secara regresi.
- **Success metric**: Single source of truth untuk logika pembersihan teks yang diimpor konsisten oleh pipeline kualitas data maupun script pemodelan.

---

### P1-3: Pembersihan Artefak Usang & Direktori Typo di Root
- **Goal**: Menghapus folder hasil typo `.kagge-outputs/`, file kosong `notebook.ipynb` (75 bytes), dan file `package-lock.json` yang tidak terpakai dari root repositori.
- **Files**:
  - `.kagge-outputs/`
  - `notebook.ipynb`
  - `package-lock.json`
  - [`.gitignore`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/.gitignore)
- **Risk**: Tidak ada risiko fungsional selama file yang dihapus bukan file data utama atau artefak penting.
- **Success metric**: Root repositori bersih dari file non-standar dan folder typo; `.gitignore` diperbarui untuk mencegah terulangnya komit file sementara.

---

### P1-4: Pembuatan Rangkaian Pengujian Otomatis (`tests/`)
- **Goal**: Mengimplementasikan suite pengujian berbasis `pytest` untuk memverifikasi logika parsing config, factory loss/trainer, kalkulasi Macro F1/ECE, dan compiler generator notebook.
- **Files**:
  - `tests/test_config.py`
  - `tests/test_metrics.py`
  - `tests/test_trainer_factory.py`
  - `tests/test_generate_notebook.py`
  - [`requirements.txt`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/requirements.txt)
- **Risk**: Overhead waktu penulisan kode uji.
- **Success metric**: Perintah `pytest` berhasil menjalankan seluruh pengujian unit dengan status 100% pass tanpa kegagalan regresi.

---

## P2 Nice to Have (Developer Experience, Ekstensibilitas & Otomatisasi)

### P2-1: Parameterisasi Kredensial & Lingkungan Kaggle
- **Goal**: Mengabstraksi nama pengguna Kaggle (`emanuelembuaijdak`) pada generator notebook agar dibaca secara dinamis melalui environment variable (`KAGGLE_USERNAME`) atau konfigurasi `.env`.
- **Files**:
  - [`tools/generate_notebook.py`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools/generate_notebook.py)
  - [`.env.example`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/.env.example)
- **Risk**: Kesalahan pembacaan environment variable dapat menghasilkan `kernel-metadata.json` dengan username kosong jika tidak disediakan nilai fallback.
- **Success metric**: Generator dapat berjalan lancar di lingkungan pengembang lain tanpa perlu mengubah kode sumber Python secara manual.

---

### P2-2: Pembuatan Unified CLI Runner (`cli.py`)
- **Goal**: Membangun satu antarmuka CLI terintegrasi yang membungkus sub-command generator notebook, pipeline kualitas data, kalibrasi ambang batas, dan verifikasi metrik.
- **Files**:
  - `cli.py`
  - [`docs/HANDOVER.md`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/docs/HANDOVER.md)
- **Risk**: Peningkatan abstraksi kode CLI.
- **Success metric**: Pengguna dapat menjalankan seluruh alur kerja repositori menggunakan satu perintah terpusat (contoh: `python cli.py compile --config ...` atau `python cli.py audit`).

---

### P2-3: Konfigurasi Otomatisasi Linter & Code Formatter
- **Goal**: Menambahkan konfigurasi linter modern (`ruff`) dan pre-commit hooks untuk menjamin kerapian sintaks, type hints, dan konsistensi format di seluruh file Python.
- **Files**:
  - `pyproject.toml`
  - `.pre-commit-config.yaml`
- **Risk**: Potensi timbulnya perubahan format baris secara massal pada riwayat git.
- **Success metric**: 100% file Python di [`src/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/src), [`tools/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/tools), dan [`quality_pipeline/`](file:///D:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/quality_pipeline) lolos pemeriksaan linter tanpa peringatan sintaks.
